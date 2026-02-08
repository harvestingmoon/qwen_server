import requests
import json
import base64
from pathlib import Path
from typing import Optional, List, Dict, Union


class QwenVLClient:
    """Client for Qwen-2-VL model via OpenAI-compatible API"""
    
    def __init__(self, base_url: str = "http://192.168.18.102:8000/v1"):
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/chat/completions"
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def chat(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        model: str = "Qwen2-VL-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Union[str, requests.Response]:
        """
        Send a chat request to Qwen-2-VL
        
        Args:
            prompt: Text prompt/question
            image_path: Local path to image file
            image_url: URL to image
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text response or streaming response object
        """
        # Build message content
        content = []
        
        # Add image if provided
        if image_path:
            image_data = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })
        elif image_url:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Build request payload
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Send request
        headers = {"Content-Type": "application/json"}
        
        if stream:
            response = requests.post(
                self.chat_url,
                json=payload,
                headers=headers,
                stream=True
            )
            response.raise_for_status()
            return response
        else:
            response = requests.post(
                self.chat_url,
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
    
    def chat_stream(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        **kwargs
    ):
        """Stream chat responses"""
        response = self.chat(
            prompt=prompt,
            image_path=image_path,
            image_url=image_url,
            stream=True,
            **kwargs
        )
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue


def main():
    """Example usage"""
    client = QwenVLClient()
    
    print("=" * 60)
    print("Qwen-2-VL Client - Example Usage")
    print("=" * 60)
    
    # Example 1: Text-only query
    print("\n[Example 1] Text-only query:")
    print("-" * 60)
    try:
        response = client.chat(
            prompt="What is the capital of France?",
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Image + Text query (from local file)
    print("\n[Example 2] Image + Text query (local file):")
    print("-" * 60)
    image_path = "example_image.jpg"  # Replace with your image path
    if Path(image_path).exists():
        try:
            response = client.chat(
                prompt="What do you see in this image?",
                image_path=image_path,
                temperature=0.7
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Image not found: {image_path}")
    
    # Example 3: Image URL + Text query
    print("\n[Example 3] Image + Text query (URL):")
    print("-" * 60)
    try:
        response = client.chat(
            prompt="Describe this image in detail.",
            image_url="https://example.com/image.jpg",  # Replace with actual URL
            temperature=0.7
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Streaming response
    print("\n[Example 4] Streaming response:")
    print("-" * 60)
    try:
        print("Response: ", end="", flush=True)
        for chunk in client.chat_stream(
            prompt="Write a short poem about AI.",
            temperature=0.8
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
