import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "models/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True
)

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map={"": 0},
    local_files_only=True,
    attn_implementation="eager",
)

model.eval()

prompt = "Explain how lithium-ion batteries work."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to("cuda")

with torch.inference_mode():
    outputs = model(**inputs)

logits = outputs.logits[:, -1, :]

print("Logits dtype:", logits.dtype)
print("Logits shape:", logits.shape)
print("NaN count:", torch.isnan(logits).sum().item())
print("Inf count:", torch.isinf(logits).sum().item())
print("Finite logits:", torch.isfinite(logits).sum().item())
print("Min logit:", logits.min().item())
print("Max logit:", logits.max().item())
print("Argmax token:", logits.argmax(dim=-1).item())

print("Top 10 token IDs:", torch.topk(logits, 10).indices[0].tolist())
