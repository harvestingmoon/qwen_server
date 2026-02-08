from io import BytesIO
import os

import requests
import torch
from PIL import Image

from vllm import LLM

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_image_from_url(url) -> Image.Image:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


if __name__ == '__main__':
    try:
        model = LLM(
            model="./Qwen3-VL-Embedding-2B-FP8",
            runner="pooling",
            max_model_len=10000,
            dtype="bfloat16",
            load_format="safetensors",
            enable_prefix_caching=False,
            gpu_memory_utilization=0.75,
        )

        image = get_image_from_url("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
        image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        inputs = [
            {
                "prompt": "A woman playing with her dog on a beach at sunset.",
            },
            {
                "prompt": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
            },
            {
                "prompt": image_placeholder,
                "multi_modal_data": {"image": image},
            },
            {
                "prompt": f"{image_placeholder}\nA woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
                "multi_modal_data": {"image": image},
            },
        ]

        outputs = model.embed(inputs)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        scores = embeddings[:2] @ embeddings[2:].T
        print(scores.tolist())
        
    finally:
        # Clean up distributed resources
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
