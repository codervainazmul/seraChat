from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import uuid
import traceback
from g4f.client import Client

app = FastAPI(title="Robust Chat API", description="Streaming Chat API with Better Error Handling")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize g4f client
client = Client()

# In-memory conversation storage
conversations: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    model: str
    prompt: str
    conversation_id: Optional[str] = None

def safe_extract_content(obj: Any, field_name: str = "content") -> str:
    """Safely extract content from any object"""
    if obj is None:
        return ""
    
    try:
        # Try direct access
        if hasattr(obj, field_name):
            value = getattr(obj, field_name)
            if value is not None:
                return str(value)
        
        # Try as dictionary
        if isinstance(obj, dict) and field_name in obj:
            return str(obj[field_name])
        
        # Try to convert the whole object
        if isinstance(obj, str):
            return obj
        
        # Try to get string representation
        return str(obj)
        
    except Exception as e:
        return f"[Content extraction error: {str(e)}]"

def debug_object_structure(obj: Any, name: str = "object") -> str:
    """Debug helper to understand object structure"""
    try:
        info = []
        info.append(f"{name} type: {type(obj)}")
        
        if hasattr(obj, '__dict__'):
            info.append(f"{name} dict: {obj.__dict__}")
        
        if hasattr(obj, 'content'):
            info.append(f"{name}.content: {obj.content} (type: {type(obj.content)})")
        
        if hasattr(obj, 'reasoning'):
            info.append(f"{name}.reasoning: {obj.reasoning} (type: {type(obj.reasoning)})")
        
        return " | ".join(info)
    except:
        return f"{name}: Could not debug"

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Robust streaming chat with better error handling"""
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
                # Send conversation ID first
                yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"
                
                # Get streaming response from g4f
                chat_completion = client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    stream=True
                )
                
                full_response = ""
                completion_count = 0
                
                # Stream the response
                for completion in chat_completion:
                    completion_count += 1
                    
                    try:
                        # Debug first few completions
                        if completion_count <= 2:
                            debug_info = {
                                "type": "debug",
                                "info": debug_object_structure(completion, "completion"),
                                "conversation_id": conversation_id
                            }
                            yield f"data: {json.dumps(debug_info)}\n\n"
                        
                        # Extract delta safely
                        delta = None
                        if hasattr(completion, 'choices') and completion.choices:
                            choice = completion.choices[0]
                            if hasattr(choice, 'delta'):
                                delta = choice.delta
                        
                        if delta is None:
                            continue
                        
                        # Handle content
                        content = safe_extract_content(delta, 'content')
                        if content and content != "[Content extraction error: 'NoneType' object has no attribute 'content']":
                            full_response += content
                            data = {
                                "type": "content",
                                "content": content,
                                "conversation_id": conversation_id,
                                "model": request.model
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                        
                        # Handle reasoning
                        if hasattr(delta, 'reasoning') and delta.reasoning is not None:
                            reasoning_content = safe_extract_content(delta.reasoning, 'content')
                            
                            # If no content field, try the whole reasoning object
                            if not reasoning_content or reasoning_content.startswith("[Content extraction error"):
                                reasoning_content = safe_extract_content(delta.reasoning)
                            
                            if reasoning_content and not reasoning_content.startswith("[Content extraction error"):
                                full_response += reasoning_content
                                data = {
                                    "type": "reasoning",
                                    "content": reasoning_content,
                                    "conversation_id": conversation_id,
                                    "model": request.model
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                        
                        # Stop after reasonable number of completions to prevent infinite loops
                        if completion_count > 1000:
                            yield f"data: {json.dumps({'type': 'warning', 'content': 'Max completions reached'})}\n\n"
                            break
                            
                    except Exception as delta_error:
                        # Send delta error but continue
                        error_data = {
                            "type": "delta_error",
                            "error": str(delta_error),
                            "completion_count": completion_count,
                            "conversation_id": conversation_id
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                        continue
                
                # Save assistant response to conversation
                if full_response:
                    conversations[conversation_id].append({"role": "assistant", "content": full_response})
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id, 'total_completions': completion_count})}\n\n"
                
            except Exception as e:
                # Send detailed error information
                error_data = {
                    "type": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "model": request.model,
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

@app.get("/")
async def root():
    return {"message": "Robust Chat API Server", "status": "running"}

@app.get("/test-model/{model}")
async def test_model(model: str):
    """Test a model without streaming to see its structure"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello, test message"}],
            stream=False
        )
        
        result = {
            "model": model,
            "response_type": str(type(response)),
            "response_dir": dir(response),
            "success": True
        }
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            message = choice.message
            result["message_type"] = str(type(message))
            result["message_dir"] = dir(message)
            
            if hasattr(message, '__dict__'):
                result["message_dict"] = str(message.__dict__)
        
        return result
        
    except Exception as e:
        return {
            "model": model,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
