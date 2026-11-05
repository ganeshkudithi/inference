import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
PAD_LENGTH = 84
MAX_NEW_TOKENS = 128

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
    dtype=torch.float32,
    local_files_only=True,
    device_map="cuda:0",
)
model.eval()

def generate(prompt_list):
    encoded = tokenizer(
        prompt_list,
        return_tensors="pt",
        padding="max_length",
        max_length=PAD_LENGTH,
        truncation=False,
    ).to("cuda:0")

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[:, PAD_LENGTH:].cpu().tolist()

    # Trim at the first actual EOS token.
    trimmed = []
    for sequence in generated:
        if tokenizer.eos_token_id in sequence:
            sequence = sequence[:sequence.index(tokenizer.eos_token_id) + 1]
        trimmed.append(sequence)

    return trimmed

def compare(reference, candidate):
    limit = min(len(reference), len(candidate))
    first_diff = next(
        (i for i in range(limit) if reference[i] != candidate[i]),
        None,
    )

    if first_diff is None and len(reference) != len(candidate):
        first_diff = limit

    return first_diff, reference == candidate

print("\nGENERATE API CONSISTENCY TEST")
print(f"Fixed input width: {PAD_LENGTH}")

references = {}
for prompt_id, prompt in prompts.items():
    references[prompt_id] = generate([prompt])[0]
    print(
        f"Reference {prompt_id}: "
        f"{len(references[prompt_id])} tokens"
    )

pairs = [
    ("short_01", "long_01"),
    ("medium_01", "long_01"),
    ("long_01", "short_01"),
]

print("\nBATCHED COMPARISONS")

for target_id, partner_id in pairs:
    outputs = generate([prompts[target_id], prompts[partner_id]])
    candidate = outputs[0]

    first_diff, exact = compare(
        references[target_id], candidate
    )

    print(
        f"{target_id} with {partner_id}: "
        f"{'MATCH' if exact else 'MISMATCH'} | "
        f"reference_length={len(references[target_id])} | "
        f"batch_length={len(candidate)} | "
        f"first_difference={first_diff}"
    )

print("\nDiagnostic complete.")
