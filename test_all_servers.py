import requests

# Server URLs
EMBEDDING_SERVER = "http://localhost:8000"
CHAT_SERVER = "http://localhost:8001"

print("=" * 60)
print("Testing Qwen VL Embedding Server (Port 8000)")
print("=" * 60)

def test_embedding_health():
    """Test embedding server health"""
    print("\n1. Testing embedding server health...")
    try:
        response = requests.get(f"{EMBEDDING_SERVER}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")


def test_text_embedding():
    """Test text-only embedding"""
    print("\n2. Testing text embedding...")
    
    payload = {
        "inputs": [
            {"prompt": "A woman playing with her dog on a beach at sunset."},
            {"prompt": "A man walking in the park."}
        ]
    }
    
    try:
        response = requests.post(f"{EMBEDDING_SERVER}/embed", json=payload)
        result = response.json()
        
        print(f"   Status: {response.status_code}")
        print(f"   Number of embeddings: {len(result['embeddings'])}")
        print(f"   Embedding dimension: {len(result['embeddings'][0])}")
        if result.get('scores'):
            print(f"   Similarity scores:")
            for i, row in enumerate(result['scores']):
                print(f"      Row {i}: {[f'{x:.4f}' for x in row]}")
    except Exception as e:
        print(f"   ERROR: {e}")


def test_image_embedding():
    """Test image embedding"""
    print("\n3. Testing image + text embedding...")
    
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    payload = {
        "inputs": [
            {"prompt": "A beach scene at sunset"},
            {
                "prompt": image_placeholder,
                "multi_modal_data": {
                    "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                }
            }
        ]
    }
    
    try:
        response = requests.post(f"{EMBEDDING_SERVER}/embed", json=payload)
        result = response.json()
        
        print(f"   Status: {response.status_code}")
        print(f"   Number of embeddings: {len(result['embeddings'])}")
        if result.get('scores'):
            print(f"   Text-Image similarity: {result['scores'][0][1]:.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")


def test_openai_embeddings():
    """Test OpenAI-compatible embeddings endpoint"""
    print("\n4. Testing OpenAI-compatible embeddings...")
    
    payload = {
        "model": "qwen3-vl-embedding",
        "input": "Hello world"
    }
    
    try:
        response = requests.post(f"{EMBEDDING_SERVER}/v1/embeddings", json=payload)
        result = response.json()
        
        print(f"   Status: {response.status_code}")
        print(f"   Embedding dimension: {len(result['data'][0]['embedding'])}")
        print(f"   Model: {result['model']}")
    except Exception as e:
        print(f"   ERROR: {e}")


print("\n" + "=" * 60)
print("Testing Qwen3-0.6B Chat Server (Port 8001)")
print("=" * 60)

def test_chat_health():
    """Test chat server health"""
    print("\n5. Testing chat server health...")
    try:
        response = requests.get(f"{CHAT_SERVER}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")


def test_chat_models():
    """Test models endpoint"""
    print("\n6. Testing models endpoint...")
    try:
        response = requests.get(f"{CHAT_SERVER}/v1/models")
        result = response.json()
        print(f"   Available models: {[m['id'] for m in result['data']]}")
    except Exception as e:
        print(f"   ERROR: {e}")


def test_chat_completion():
    """Test chat completion"""
    print("\n7. Testing chat completion...")
    
    payload = {
        "model": "qwen3-0.6b-fp8",
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer in one sentence."}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(f"{CHAT_SERVER}/v1/chat/completions", json=payload)
        result = response.json()
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            choice = result['choices'][0]
            print(f"   Response: {choice['message']['content']}")
            print(f"   Usage: {result['usage']}")
        else:
            print(f"   Error: {result}")
    except Exception as e:
        print(f"   ERROR: {e}")


def test_chat_multi_turn():
    """Test multi-turn conversation"""
    print("\n8. Testing multi-turn conversation...")
    
    payload = {
        "model": "qwen3-0.6b-fp8",
        "messages": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What's my name?"}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(f"{CHAT_SERVER}/v1/chat/completions", json=payload)
        result = response.json()
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            choice = result['choices'][0]
            print(f"   Response: {choice['message']['content']}")
        else:
            print(f"   Error: {result}")
    except Exception as e:
        print(f"   ERROR: {e}")


if __name__ == "__main__":
    try:
        # Test Embedding Server
        test_embedding_health()
        test_text_embedding()
        test_image_embedding()
        test_openai_embeddings()
        
        # Test Chat Server
        test_chat_health()
        test_chat_models()
        test_chat_completion()
        test_chat_multi_turn()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
