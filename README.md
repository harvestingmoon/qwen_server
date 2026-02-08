# Qwen3-VL Embedding Server

A FastAPI-based embedding server for Qwen3-VL-Embedding-2B-FP8 model, supporting both text and image embeddings.

## Features

- Text embedding generation
- Image + text multimodal embedding
- RESTful API with FastAPI
- Network accessible (LAN/WiFi)
- Automatic similarity scoring

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- WSL2 (if running on Windows)

## Setup

### 1. Clone the Model

Download the Qwen3-VL-Embedding-2B-FP8 model to the project directory:

```bash
cd qwen_server
git clone https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B-FP8
cd Qwen3-VL-Embedding-2B-FP8
git lfs pull  # Pull large files
```

**Note:** The `Qwen3-VL-Embedding-2B-FP8` folder is excluded from git via `.gitignore`.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For WSL/Linux, install build tools:
```bash
sudo apt update
sudo apt install -y build-essential
```

## Usage

### Start the Server

```bash
python server.py
```


### API Endpoints

#### POST `/embed`

Generate embeddings for text and/or images.

**Request:**
```json
{
  "inputs": [
    {
      "prompt": "A woman playing with her dog on a beach at sunset."
    },
    {
      "prompt": "<|vision_start|><|image_pad|><|vision_end|>",
      "multi_modal_data": {
        "image_url": "https://example.com/image.jpg"
      }
    }
  ]
}
```

**Response:**
```json
{
  "embeddings": [[...], [...]],
  "scores": [[1.0, 0.85], [0.85, 1.0]]
}
```

#### GET `/health`

Health check endpoint.

#### GET `/`

Server info and available endpoints.

## Testing

Run the test client:

```bash
python client_test.py
```

Or test from another device:

```python
import requests

response = requests.post(
    "http://192.168.x.xxx:8000/embed",
    json={
        "inputs": [
            {"prompt": "Your text here"}
        ]
    }
)
print(response.json())
```

## Configuration

Edit model parameters in `server.py`:

```python
model = LLM(
    model="./Qwen3-VL-Embedding-2B-FP8",
    max_model_len=10000,          # Adjust based on GPU memory
    gpu_memory_utilization=0.75,   # 0.5-0.9 depending on available VRAM
)
```

## Files

- `server.py` - FastAPI server
- `pipeline.py` - Standalone embedding script
- `client_test.py` - Test client
- `requirements.txt` - Python dependencies
- `.gitignore` - Excludes model folder

## Troubleshooting

### Out of Memory
- Reduce `max_model_len` (e.g., 8000, 6000)
- Lower `gpu_memory_utilization` (e.g., 0.6, 0.5)

### Can't Connect from Other Device
- Check firewall settings
- Ensure both devices are on same network
- Use the IP shown when server starts (not 127.0.0.1)

### Model Not Loading
- Verify model files exist in `./Qwen3-VL-Embedding-2B-FP8`
- Run `git lfs pull` in model directory
- Check GPU availability with `nvidia-smi`

## License

See model license at [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B-FP8)
