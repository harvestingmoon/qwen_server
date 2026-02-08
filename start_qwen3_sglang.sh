#!/bin/bash

# Start SGLang server for Qwen3-0.6B-FP8
# Using Triton backend to avoid CUDA compilation

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B-FP8 \
    --host 0.0.0.0 \
    --port 8001 \
    --mem-fraction-static 0.65 \
    --attention-backend triton \
    --disable-cuda-graph \
    --trust-remote-code
