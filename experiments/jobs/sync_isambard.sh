# Patch the installation of torch in the compute node (so that it downloads the CUDA version).
# The login node doesn't have the CUDA libraries available.
uv pip install torch torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/cu124
uv pip install triton
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/0.10.2/vllm
uv pip install -U "vllm>=0.10.0,<=0.11.2" --torch-backend=auto --index-url https://wheels.vllm.ai/0.11.2/vllm
uv pip install -e ./lighteval
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
