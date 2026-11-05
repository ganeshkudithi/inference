
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
PRECISION = torch.float32

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
]

formatted = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in prompts
]

# First three tokens shared by both executions.
shared_tokens = [12, 10167, 4756]


def get_logits(texts):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    )

    encoded = {
        k: v.to("cuda:0")
        for k, v in encoded.items()
    }

    extra_tokens = torch.tensor(
        [shared_tokens] * len(texts),
        device="cuda:0",
        dtype=encoded["input_ids"].dtype,
    )

    encoded["input_ids"] = torch.cat(
        [encoded["input_ids"], extra_tokens],
        dim=1,
    )

    encoded["attention_mask"] = torch.cat(
        [
            encoded["attention_mask"],
            torch.ones(
                (len(texts), len(shared_tokens)),
                device="cuda:0",
                dtype=encoded["attention_mask"].dtype,
            ),
        ],
        dim=1,
    )

    with torch.inference_mode():
        output = model(**encoded, use_cache=False)

    return output.logits[:, -1, :].float().cpu()


alone = get_logits([formatted[0]])[0]
batched = get_logits(formatted)[0]

difference = torch.abs(alone - batched)

alone_top = torch.topk(alone, 5)
batch_top = torch.topk(batched, 5)

print("\n--- Step 4 logit comparison ---")
print("Maximum absolute difference:", difference.max().item())
print("Mean absolute difference:", difference.mean().item())

print("\nTop 5 when alone:")
for token, score in zip(alone_top.indices, alone_top.values):
    print(
        token.item(),
        repr(tokenizer.decode([token.item()])),
        score.item(),
    )

print("\nTop 5 when batched:")
for token, score in zip(batch_top.indices, batch_top.values):
    print(
        token.item(),
        repr(tokenizer.decode([token.item()])),
        score.item(),
    )

print("\nTop-1 agreement:",
      alone_top.indices[0].item() == batch_top.indices[0].item())

del model
torch.cuda.empty_cache()

