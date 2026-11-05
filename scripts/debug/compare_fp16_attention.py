import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "models/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True
)

inputs = tokenizer(
    "Explain how lithium-ion batteries work.",
    return_tensors="pt"
).to("cuda")

for attention in ["eager", "sdpa"]:

    print("\nAttention:", attention)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation=attention,
    )

    model.eval()

    with torch.inference_mode():
        output = model(**inputs)

    logits = output.logits

    print("Logits finite:", torch.isfinite(logits).all().item())
    print("NaN count:", torch.isnan(logits).sum().item())
    print("Inf count:", torch.isinf(logits).sum().item())

    del model
    torch.cuda.empty_cache()
