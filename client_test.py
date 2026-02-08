import requests
import json

# Replace with your server's IP address
SERVER_URL = "http://localhost:8000"

def test_text_embedding():
    """Test text-only embedding"""
    print("Testing text embedding...")
    
    payload = {
        "inputs": [
            {"prompt": "A woman playing with her dog on a beach at sunset."},
            {"prompt": "A man walking in the park."}
        ]
    }
    
    response = requests.post(f"{SERVER_URL}/embed", json=payload)
    result = response.json()
    
    print(f"Status: {response.status_code}")
    print(f"Number of embeddings: {len(result['embeddings'])}")
    print(f"Embedding dimension: {len(result['embeddings'][0])}")
    if result['scores']:
        print(f"Similarity scores:\n{result['scores']}")
    print()


def test_image_embedding():
    """Test image + text embedding"""
    print("Testing image + text embedding...")
    
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    payload = {
        "inputs": [
            {"prompt": "A woman playing with her dog on a beach at sunset."},
            {
                "prompt": image_placeholder,
                "multi_modal_data": {
                    "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                }
            }
        ]
    }
    
    response = requests.post(f"{SERVER_URL}/embed", json=payload)
    result = response.json()
    
    print(f"Status: {response.status_code}")
    print(f"Number of embeddings: {len(result['embeddings'])}")
    if result['scores']:
        print(f"Similarity scores:\n{result['scores']}")
    print()


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{SERVER_URL}/health")
    print(f"Health status: {response.json()}")
    print()


if __name__ == "__main__":
    print(f"Testing server at {SERVER_URL}\n")
    
    try:
        test_health()
        test_text_embedding()
        test_image_embedding()
        print("All tests completed!")
    except Exception as e:
        print(f"Error: {e}")
