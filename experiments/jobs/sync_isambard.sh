# Patch the installation of torch in the compute node (so that it downloads the CUDA version).
# The login node doesn't have the CUDA libraries available.
uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --reinstall triton
uv pip install --reinstall -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/0.10.2/vllm
python -c "import torch; print(torch.cuda.is_available())"
