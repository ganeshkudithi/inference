
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


def get_next_token_logits(input_texts):
    encoded = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
    )
    encoded = {k: v.to("cuda:0") for k, v in encoded.items()}

    with torch.inference_mode():
        output = model(**encoded, use_cache=False)

    return output.logits[:, -1, :].float().cpu()


def inspect_generation(texts, steps=5):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    )
    encoded = {k: v.to("cuda:0") for k, v in encoded.items()}

    generated = []

    for step in range(steps):
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)

        logits = output.logits[:, -1, :].float().cpu()
        top_tokens = torch.argmax(logits, dim=-1)

        generated.append(top_tokens.tolist())

        # Append predicted tokens for the next decoding step.
        next_tokens = top_tokens.to("cuda:0").unsqueeze(1)

        encoded["input_ids"] = torch.cat(
            [encoded["input_ids"], next_tokens],
            dim=1,
        )

        encoded["attention_mask"] = torch.cat(
            [
                encoded["attention_mask"],
                torch.ones(
                    (len(texts), 1),
                    dtype=encoded["attention_mask"].dtype,
                    device="cuda:0",
                ),
            ],
            dim=1,
        )

    return generated


# Short prompt alone versus short prompt in a batch.
alone = inspect_generation([formatted[0]])
batched = inspect_generation(formatted)

print("\n--- First five decoding steps ---")

for step in range(5):
    token_alone = alone[step][0]
    token_batch = batched[step][0]

    print(f"\nStep {step + 1}")
    print("Alone:", token_alone, tokenizer.decode([token_alone]))
    print("Batch:", token_batch, tokenizer.decode([token_batch]))
    print("Match:", token_alone == token_batch)

del model
torch.cuda.empty_cache()

