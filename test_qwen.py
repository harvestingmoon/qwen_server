#!/usr/bin/env python3
"""Quick test script for Qwen-2-VL client"""

from cross_client import QwenVLClient

# Initialize client
client = QwenVLClient(base_url="http://192.168.18.102:8000/v1")

# Test 1: Simple text query
print("Testing text-only query...")
try:
    response = client.chat(
        prompt="Hello! Can you introduce yourself?",
        temperature=0.7,
        max_tokens=200
    )
    print(f"✓ Response: {response}\n")
except Exception as e:
    print(f"✗ Error: {e}\n")

# Test 2: Text query with streaming
print("Testing streaming response...")
try:
    print("✓ Response: ", end="", flush=True)
    for chunk in client.chat_stream(
        prompt="Count from 1 to 5.",
        temperature=0.7,
        max_tokens=100
    ):
        print(chunk, end="", flush=True)
    print("\n")
except Exception as e:
    print(f"✗ Error: {e}\n")

# Test 3: Image analysis (uncomment and add image path)
# print("Testing image analysis...")
# try:
#     response = client.chat(
#         prompt="What objects do you see in this image?",
#         image_path="path/to/your/image.jpg",
#         temperature=0.7
#     )
#     print(f"✓ Response: {response}\n")
# except Exception as e:
#     print(f"✗ Error: {e}\n")

print("Tests complete!")
