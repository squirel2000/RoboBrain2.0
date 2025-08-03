import asyncio
import time
import uuid
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# We will import the inference logic from the existing script
from inference import SimpleInference

# --- Data Models for OpenAI Compatibility ---
class Message(BaseModel):
    role: str
    content: Any  # Allow content to be a string or a list

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage


# --- FastAPI Application ---
app = FastAPI()

# Load the RoboBrain model once when the server starts
print("Initializing RoboBrain 2.0 model...")
model_inference = SimpleInference("BAAI/RoboBrain2.0-7B")
print("Model loaded successfully.")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint for chat completions.
    """
    # The RoboOS Master sends the entire conversation history.
    # The last message is the current user prompt.
    if not request.messages:
        return JSONResponse(status_code=400, content={"error": "messages list is empty"})

    last_message = request.messages[-1]
    if isinstance(last_message.content, list):
        # Handle the case where content is a list of dictionaries
        user_prompt = "".join(item.get('text', '') for item in last_message.content if item.get('type') == 'text')
    else:
        # Handle the case where content is a simple string
        user_prompt = last_message.content

    print(f"Received prompt for model: {user_prompt}")

    # --- Real Inference Call ---
    # Now we call the actual inference method from our modified script.
    # We pass the user prompt and set image to None.
    try:
        result = model_inference.inference(
            text=user_prompt, 
            image=None,  # Passing None for text-only inference
            task="general",
            enable_thinking=False,  # The Master planner usually expects a direct answer
            do_sample=request.temperature > 0,
            temperature=request.temperature,
        )

        response_content = result["answer"]
        print(f"Model generated answer: {response_content}")

    except Exception as e:
        print(f"Error during model inference: {e}")
        return JSONResponse(status_code=500, content={"error": "Model inference failed"})


    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=Message(role="assistant", content=response_content)
            )
        ],
        usage=Usage()
    )

if __name__ == "__main__":
    print("Starting RoboBrain 2.0 API Server...")
    # The RoboOS config points to port 8001
    uvicorn.run(app, host="127.0.0.1", port=8001)