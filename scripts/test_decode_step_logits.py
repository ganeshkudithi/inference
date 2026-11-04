import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
PAD_LENGTH = 84
MAX_STEPS = 80

torch.manual_seed(42)

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

eos_id = tokenizer.eos_token_id

# Tokenize each prompt to the same initial width.
encoded_prompts = {}
for prompt_id, prompt in prompts.items():
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=PAD_LENGTH,
        truncation=False,
    )
    encoded_prompts[prompt_id] = {
        "ids": encoded["input_ids"][0].tolist(),
        "mask": encoded["attention_mask"][0].tolist(),
    }

def forward_logits(batch_ids, batch_masks):
    input_ids = torch.tensor(batch_ids, dtype=torch.long, device="cuda:0")
    attention_mask = torch.tensor(
        batch_masks, dtype=torch.long, device="cuda:0"
    )

    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

    return output.logits[:, -1, :].float().cpu()

def make_sequence(prompt_id, generated, total_steps=None):
    base = encoded_prompts[prompt_id]

    ids = base["ids"] + generated
    mask = base["mask"] + [1] * len(generated)

    if total_steps is not None:
        missing = total_steps - len(generated)

        if missing > 0:
            ids += [tokenizer.pad_token_id] * missing
            mask += [1] * missing

    return ids, mask

# Generate individual greedy reference sequences.
references = {}

print("Generating individual reference sequences...")
for prompt_id in prompts:
    generated = []

    for step in range(MAX_STEPS):
        ids, mask = make_sequence(prompt_id, generated)
        logits = forward_logits([ids], [mask])
        next_token = logits.argmax(dim=-1).item()
        generated.append(next_token)

        if next_token == eos_id:
            break

    references[prompt_id] = generated
    print(f"  {prompt_id}: {len(generated)} generated tokens")

# Compare each target's next-token prediction in individual
# and paired execution, using identical generated prefixes.
print("\nDECODING-STEP COMPARISON")

pairs = [
    ("short_01", "long_01"),
    ("medium_01", "long_01"),
    ("long_01", "short_01"),
]

for target_id, partner_id in pairs:
    target_ref = references[target_id]
    partner_ref = references[partner_id]

    print(f"\n{target_id} paired with {partner_id}")

    first_mismatch = None
    mismatch_count = 0

    for step in range(min(MAX_STEPS, len(target_ref))):
        target_ids, target_mask = make_sequence(
            target_id, target_ref[:step], step
        )
        partner_ids, partner_mask = make_sequence(
            partner_id, partner_ref[:step], step
        )

        individual_logits = forward_logits(
            [target_ids], [target_mask]
        )
        batched_logits = forward_logits(
            [target_ids, partner_ids],
            [target_mask, partner_mask],
        )[0:1]

        individual_token = individual_logits.argmax(dim=-1).item()
        batched_token = batched_logits.argmax(dim=-1).item()

        if individual_token != batched_token:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = step

    print(f"  First differing prediction step: {first_mismatch}")
    print(f"  Total differing steps: {mismatch_count}")

print("\nDiagnostic complete.")
