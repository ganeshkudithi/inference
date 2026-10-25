
import os
import platform
import subprocess
import sys

import torch
import transformers

print("=== InferX Environment ===")
print("Python:", sys.version.split()[0])
print("OS:", platform.platform())
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Slurm job:", os.getenv("SLURM_JOB_ID", "Not allocated"))
print("CUDA_VISIBLE_DEVICES:",
      os.getenv("CUDA_VISIBLE_DEVICES", "Not set"))

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {gpu.name}")
        print(f"VRAM: {gpu.total_memory / 1024**3:.2f} GiB")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

try:
    subprocess.run(["nvidia-smi"], check=False)
except FileNotFoundError:
    print("nvidia-smi is unavailable in this environment.")

