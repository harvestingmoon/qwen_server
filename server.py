from io import BytesIO
import os
from typing import List, Optional, Union
import base64

import requests
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from vllm import LLM

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="Qwen3-VL Embedding Server")

# Global model instance
model = None


class EmbeddingRequest(BaseModel):
    inputs: List[dict]
    

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    scores: Optional[List[List[float]]] = None


# OpenAI-compatible models
class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]


class OpenAIEmbeddingRequest(BaseModel):
    model: str = "qwen3-vl-embedding"
    input: Union[str, List[str], List[dict]]
    encoding_format: Optional[str] = "float"


class OpenAIEmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIEmbeddingData]
    model: str
    usage: dict


def get_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


def get_image_from_base64(base64_str: str) -> Image.Image:
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    return img


@app.on_event("startup")
async def startup_event():
    global model
    print("Loading model...")
    model = LLM(
        model="./Qwen3-VL-Embedding-2B-FP8",
        runner="pooling",
        max_model_len=10000,
        dtype="bfloat16",
        load_format="safetensors",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.75,
    )
    print("Model loaded successfully!")


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    """
    Generate embeddings for text and/or image inputs.
    
    Example request:
    {
        "inputs": [
            {"prompt": "A woman playing with her dog"},
            {
                "prompt": "<|vision_start|><|image_pad|><|vision_end|>",
                "multi_modal_data": {
                    "image_url": "https://example.com/image.jpg"
                }
            }
        ]
    }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process inputs
        processed_inputs = []
        image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        
        for inp in request.inputs:
            processed_inp = {"prompt": inp.get("prompt", "")}
            
            # Handle multi-modal data
            if "multi_modal_data" in inp:
                mm_data = inp["multi_modal_data"]
                
                # Handle image from URL
                if "image_url" in mm_data:
                    processed_inp["multi_modal_data"] = {
                        "image": get_image_from_url(mm_data["image_url"])
                    }
                # Handle image from base64
                elif "image_base64" in mm_data:
                    processed_inp["multi_modal_data"] = {
                        "image": get_image_from_base64(mm_data["image_base64"])
                    }
                # Handle direct image object (if provided)
                elif "image" in mm_data:
                    processed_inp["multi_modal_data"] = mm_data
            
            processed_inputs.append(processed_inp)
        
        # Generate embeddings
        outputs = model.embed(processed_inputs)
        embeddings = [o.outputs.embedding for o in outputs]
        
        # Calculate similarity scores if there are multiple inputs
        scores = None
        if len(embeddings) > 1:
            embeddings_tensor = torch.tensor(embeddings)
            # Calculate pairwise cosine similarity
            scores_tensor = embeddings_tensor @ embeddings_tensor.T
            scores = scores_tensor.tolist()
        
        return EmbeddingResponse(embeddings=embeddings, scores=scores)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def openai_embeddings(request: OpenAIEmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.
    
    Example:
    {
        "model": "qwen3-vl-embedding",
        "input": "Your text here"
    }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to list format
        if isinstance(request.input, str):
            inputs = [request.input]
        elif isinstance(request.input, list):
            inputs = request.input
        else:
            inputs = [str(request.input)]
        
        # Process inputs
        processed_inputs = []
        for inp in inputs:
            if isinstance(inp, str):
                processed_inputs.append({"prompt": inp})
            elif isinstance(inp, dict):
                # Handle dict format with potential multimodal data
                processed_inputs.append(inp)
        
        # Generate embeddings
        outputs = model.embed(processed_inputs)
        embeddings = [o.outputs.embedding for o in outputs]
        
        # Format response in OpenAI style
        data = [
            OpenAIEmbeddingData(
                embedding=emb,
                index=idx
            )
            for idx, emb in enumerate(embeddings)
        ]
        
        return OpenAIEmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": sum(len(inp.get("prompt", "").split()) for inp in processed_inputs),
                "total_tokens": sum(len(inp.get("prompt", "").split()) for inp in processed_inputs)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen3-VL Embedding Server",
        "endpoints": {
            "/embed": "POST - Generate embeddings",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


if __name__ == "__main__":
    import socket
    
    # Get local IP address - try multiple methods
    def get_local_ip():
        try:
            # Create a socket to get the actual network IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            # Fallback method
            try:
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                if not local_ip.startswith("127."):
                    return local_ip
            except:
                pass
            return "Unable to determine"
    
    local_ip = get_local_ip()
    
    print(f"\nStarting server...")
    print(f"Local IP: {local_ip}")
    print(f"Access from other devices: http://{local_ip}:8000")
    print(f"Access locally: http://localhost:8000")
    print(f"API docs: http://localhost:8000/docs\n")
    
    # Bind to 0.0.0.0 to allow external access
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
