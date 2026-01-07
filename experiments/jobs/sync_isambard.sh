# Patch the installation of torch in the compute node (so that it downloads the CUDA version).
# The login node doesn't have the CUDA libraries available.
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install triton
uv pip install -U "vllm>=0.10.0,<0.12.0" --torch-backend=auto --extra-index-url https://wheels.vllm.ai/0.10.2/vllm
uv pip install -e ./lighteval
uv pip install ctranslate2
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
python -c "import vllm; print(vllm.__version__)"
