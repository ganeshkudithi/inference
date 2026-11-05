import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
MAX_INPUT_LENGTH = 84

torch.set_grad_enabled(False)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True
)
tokenizer.padding_side = "left"

items = [
    json.loads(line)
    for line in Path("workloads/prompts.jsonl").read_text().splitlines()
    if line.strip()
]

prompts = {}
for item in items:
    messages = [{"role": "user", "content": item["prompt"]}]
    prompts[item["id"]] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cuda:0",
)
model.eval()

def get_last_logits(prompt_list):
    encoded = tokenizer(
        prompt_list,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_INPUT_LENGTH,
        truncation=False,
    ).to("cuda:0")

    # Explicit position IDs based on the attention mask
    position_ids = encoded["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(encoded["attention_mask"] == 0, 1)

    with torch.inference_mode():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            position_ids=position_ids,
            use_cache=False,
        )

    # Left padding means the final real prompt token is at index 83.
    return output.logits[:, -1, :].float().cpu()

print("\nPREFILL LOGIT COMPARISON")
print("All inputs padded to 84 tokens")
print("Comparing the first next-token prediction\n")

for prompt_id, prompt in prompts.items():
    reference = get_last_logits([prompt])

    ref_top = reference.argmax(dim=-1).item()
    print(f"{prompt_id}:")
    print(f"  Individual top token ID: {ref_top}")

    for other_id, other_prompt in prompts.items():
        if prompt_id == other_id:
            continue

        candidate = get_last_logits([prompt, other_prompt])[0:1]

        candidate_top = candidate.argmax(dim=-1).item()
        max_diff = (reference - candidate).abs().max().item()
        mean_diff = (reference - candidate).abs().mean().item()

        print(
            f"  With {other_id}: "
            f"top_token={candidate_top}, "
            f"same_top1={ref_top == candidate_top}, "
            f"max_logit_diff={max_diff:.6f}, "
            f"mean_logit_diff={mean_diff:.6f}"
        )

print("\nDiagnostic complete.")
