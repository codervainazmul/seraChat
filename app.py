from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import uuid
from g4f.client import Client

app = FastAPI(title="Chat API", description="Streaming Chat API with Conversation Memory")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize g4f client
client = Client()

# In-memory conversation storage (use Redis/Database in production)
conversations: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    model: str
    prompt: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    model: str

@app.get("/")
async def root():
    return {
        "message": "Chat API Server", 
        "endpoints": {
            "/chat/stream": "POST - Stream chat responses",
            "/chat": "POST - Regular chat responses", 
            "/conversations/{conversation_id}": "GET - Get conversation history",
            "/conversations": "GET - List all conversations"
        }
    }

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Stream chat responses with conversation memory"""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get or create conversation history
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Add user message to conversation
        conversations[conversation_id].append({"role": "user", "content": request.prompt})
        
        # Prepare messages for API
        messages = conversations[conversation_id].copy()
        
        def generate_response():
            try:
                # Get streaming response from g4f
                chat_completion = client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    stream=True
                )
                
                full_response = ""
                
                # Send conversation ID first
                yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"
                
                # Stream the response
                for completion in chat_completion:
                    delta = completion.choices[0].delta
                    
                    # Handle regular content
                    content = getattr(delta, 'content', None) or ""
                    
                    # Handle reasoning models (deepseek-r1, etc.)
                    reasoning = getattr(delta, 'reasoning', None)
                    
                    if reasoning:
                        # If reasoning is an object, extract content
                        if hasattr(reasoning, 'content'):
                            reasoning_content = reasoning.content or ""
                        else:
                            reasoning_content = str(reasoning) if reasoning else ""
                        
                        if reasoning_content:
                            full_response += reasoning_content
                            data = {
                                "type": "reasoning",
                                "content": reasoning_content,
                                "conversation_id": conversation_id,
                                "model": request.model
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                    
                    if content:
                        full_response += content
                        data = {
                            "type": "content",
                            "content": content,
                            "conversation_id": conversation_id,
                            "model": request.model
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                
                # Save assistant response to conversation
                if full_response:
                    conversations[conversation_id].append({"role": "assistant", "content": full_response})
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
                
            except Exception as e:
                error_data = {
                    "type": "error",
                    "error": str(e),
                    "conversation_id": conversation_id
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def regular_chat(request: ChatRequest):
    """Regular chat endpoint (non-streaming) with conversation memory"""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Get or create conversation history
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Add user message to conversation
        conversations[conversation_id].append({"role": "user", "content": request.prompt})
        
        # Prepare messages for API
        messages = conversations[conversation_id].copy()
        
        # Get response from g4f (non-streaming)
        chat_completion = client.chat.completions.create(
            model=request.model,
            messages=messages,
            stream=False
        )
        
        message = chat_completion.choices[0].message
        response_content = ""
        
        # Handle regular content
        if hasattr(message, 'content') and message.content:
            response_content += message.content
        
        # Handle reasoning models
        if hasattr(message, 'reasoning') and message.reasoning:
            reasoning = message.reasoning
            if hasattr(reasoning, 'content'):
                response_content += reasoning.content or ""
            else:
                response_content += str(reasoning)
        
        # Save assistant response to conversation
        conversations[conversation_id].append({"role": "assistant", "content": response_content})
        
        return ChatResponse(
            conversation_id=conversation_id,
            message=response_content,
            model=request.model
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history by ID"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id],
        "message_count": len(conversations[conversation_id])
    }

@app.get("/conversations")
async def list_conversations():
    """List all conversation IDs"""
    return {
        "conversations": [
            {
                "conversation_id": conv_id,
                "message_count": len(messages),
                "last_message": messages[-1]["content"][:100] + "..." if messages else ""
            }
            for conv_id, messages in conversations.items()
        ],
        "total": len(conversations)
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    return {"message": f"Conversation {conversation_id} deleted successfully"}

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": [
            "gpt-4",
            "gpt-4-turbo", 
            "gpt-3.5-turbo",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "gemini-pro",
            "llama-2-70b-chat"
        ],
        "note": "Model availability depends on g4f provider status"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
