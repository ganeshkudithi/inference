
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
PRECISION = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=PRECISION,
    device_map={"": 0},
    local_files_only=True,
    attn_implementation="sdpa",
)
model.eval()

prompts = [
    "Explain GPU inference optimization in three concise bullet points.",
    "Explain how mixed precision, quantization, and KV caching improve large language model inference.",
    "Write a detailed technical explanation of large language model inference, including tokenization, prefill, decoding, attention, KV cache, and batching.",
]

formatted = []
for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    formatted.append(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def get_last_logits(texts):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    )
    encoded = {k: v.to("cuda:0") for k, v in encoded.items()}

    with torch.inference_mode():
        output = model(**encoded)

    # Last non-padding token for every sequence.
    # With left padding, every prompt ends at the final sequence position.
    logits = output.logits[:, -1, :]

    return logits.float().cpu()


# Use short prompt as the reference.
reference = get_last_logits([formatted[0]])[0]

print("\nReference: short prompt alone")

for batch_size in [2, 4]:
    batch = [formatted[0]] + [
        formatted[i % len(formatted)]
        for i in range(1, batch_size)
    ]

    candidate = get_last_logits(batch)[0]

    difference = torch.abs(reference - candidate)

    ref_top5 = torch.topk(reference, 5).indices
    batch_top5 = torch.topk(candidate, 5).indices

    ref_top1 = ref_top5[0].item()
    batch_top1 = batch_top5[0].item()

    ref_margin = (
        torch.topk(reference, 2).values[0]
        - torch.topk(reference, 2).values[1]
    ).item()

    batch_margin = (
        torch.topk(candidate, 2).values[0]
        - torch.topk(candidate, 2).values[1]
    ).item()

    print(f"\n========== Batch size {batch_size} ==========")
    print("Maximum absolute difference:", difference.max().item())
    print("Mean absolute difference:", difference.mean().item())
    print("Top-1 agreement:", ref_top1 == batch_top1)
    print("Top-5 overlap:", len(set(ref_top5.tolist()) & set(batch_top5.tolist())))
    print("Reference top-1 margin:", ref_margin)
    print("Batch top-1 margin:", batch_margin)
    print("Reference token:", tokenizer.decode([ref_top1]))
    print("Batch token:", tokenizer.decode([batch_top1]))

del model
torch.cuda.empty_cache()

