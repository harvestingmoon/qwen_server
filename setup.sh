conda create -n qwen_vl python=3.12 -y
conda activate qwen_vl
pip install -r requirements.txt
git clone  https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B-FP8
cd Qwen3-VL-Embedding-2B-FP8
git lfs install
git lfs pull
cd ..
python qwen_server.py --model_path ./Qwen3-VL-Embedding-2B