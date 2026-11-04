import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
MAX_INPUT_LENGTH = 84
MAX_NEW_TOKENS = 128

# Load prompts
items = [
    json.loads(line)
    for line in Path("workloads/prompts.jsonl").read_text().splitlines()
    if line.strip()
]

prompts = {}
for item in items:
    messages = [{"role": "user", "content": item["prompt"]}]
    prompts[item["id"]] = tokenizer_text = None

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True
)
tokenizer.padding_side = "left"

for item in items:
    messages = [{"role": "user", "content": item["prompt"]}]
    prompts[item["id"]] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cuda:0",
)
model.eval()

def generate(prompt_ids):
    encoded = tokenizer(
        prompt_ids,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_INPUT_LENGTH,
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

    return output[:, MAX_INPUT_LENGTH:].cpu()

def compare(name, reference, candidate):
    same = torch.equal(reference, candidate)
    ref_len = reference.shape[1]
    cand_len = candidate.shape[1]

    first_diff = None
    for i in range(min(ref_len, cand_len)):
        if reference[0, i] != candidate[0, i]:
            first_diff = i
            break

    if first_diff is None and ref_len != cand_len:
        first_diff = min(ref_len, cand_len)

    print(
        f"{name}: {'MATCH' if same else 'MISMATCH'} | "
        f"reference_tokens={ref_len} | candidate_tokens={cand_len} | "
        f"first_difference={first_diff}"
    )

# Reference: each prompt generated individually, padded to 84 tokens
references = {}
for prompt_id, prompt in prompts.items():
    references[prompt_id] = generate([prompt])

print("\nFIXED-PADDING COMPARISON")
print(f"All inputs padded to {MAX_INPUT_LENGTH} tokens")

# Compare each prompt against its output when paired with another prompt.
for prompt_id, prompt in prompts.items():
    for other_id, other_prompt in prompts.items():
        if prompt_id == other_id:
            continue

        batch_output = generate([prompt, other_prompt])
        compare(
            f"{prompt_id} paired with {other_id}",
            references[prompt_id],
            batch_output[0:1],
        )

print("\nDiagnostic complete.")
