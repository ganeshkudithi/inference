
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(
            "VRAM (GiB):",
            round(
                torch.cuda.get_device_properties(i).total_memory / 1024**3,
                2,
            ),
        )

