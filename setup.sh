pip install uv
uv venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip uninstall pynvml
uv pip install nvidia-ml-py

# download dataset
mkdir -p data
mkdir -p data/video_reason
if [ ! -d "data/video_reason/Video-Holmes" ]; then
    huggingface-cli download --repo-type dataset --resume-download k-nick/video_reason --local-dir data/video_reason/Video-Holmes
fi

cp decompress.py data/video_reason/Video-Holmes
cd data/video_reason/Video-Holmes
python decompress.py
cd ../../../

chmod a+x scripts/convert.sh
./scripts/convert.sh

# download model weights
mkdir -p model_weights
if [ ! -d "model_weights/Qwen2.5-VL-7B-Instruct" ]; then
    huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir model_weights/Qwen2.5-VL-7B-Instruct
fi

if [ ! -d "model_weights/ft_coldstart" ]; then
    huggingface-cli download --resume-download k-nick/ft_coldstart --local-dir model_weights/ft_coldstart
fi
