
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
PROMPTS_PATH = "workloads/prompts.jsonl"
PRECISION = torch.bfloat16
MAX_NEW_TOKENS = 128
REQUEST_COUNT = 24
BATCH_SIZES = [1, 2, 4, 8]

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=False,
)
tokenizer.padding_side = "left"

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=PRECISION,
    device_map={"": 0},
    local_files_only=True,
    trust_remote_code=False,
    attn_implementation="sdpa",
)
model.eval()
print("\n--- Token configuration ---")
print("Tokenizer EOS:", tokenizer.eos_token_id)
print("Tokenizer PAD:", tokenizer.pad_token_id)
print("Model EOS:", model.generation_config.eos_token_id)
print("Model PAD:", model.generation_config.pad_token_id)
# Reproduce the validator's exact prompt formatting.
source_prompts = [
    json.loads(line)
    for line in Path(PROMPTS_PATH).read_text().splitlines()
    if line.strip()
]

formatted = []
for item in source_prompts:
    messages = [{"role": "user", "content": item["prompt"]}]

    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = item["prompt"]

    formatted.append({
        "id": item["id"],
        "text": text,
    })

requests = [
    {
        "request_index": i,
        "prompt_id": formatted[i % len(formatted)]["id"],
        "text": formatted[i % len(formatted)]["text"],
    }
    for i in range(REQUEST_COUNT)
]


def generate_batch(batch):
    encoded = tokenizer(
        [item["text"] for item in batch],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    encoded = {k: v.to("cuda:0") for k, v in encoded.items()}

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.generation_config.eos_token_id,
        )
    prompt_length = encoded["input_ids"].shape[1]
    generated = output[:, prompt_length:].cpu()

    # Remove padding tokens generated after each request reaches EOS.
    eos_token_id = tokenizer.eos_token_id
    trimmed_outputs = []

    for sequence in generated:
        eos_positions = (sequence == eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_positions) > 0:
            sequence = sequence[:eos_positions[0].item() + 1]

        trimmed_outputs.append(sequence)

    return trimmed_outputs
    # prompt_length = encoded["input_ids"].shape[1]

    # return [
    #     output[i, prompt_length:].cpu()
    #     for i in range(len(batch))
    # ]


# Generate a batch-size-1 reference for every request.
print("Generating individual references...")
references = []

for request in requests:
    references.append(generate_batch([request])[0])

for batch_size in BATCH_SIZES:
    print(f"\n========== Batch size {batch_size} ==========")

    exact_matches = 0
    total = 0

    for start in range(0, REQUEST_COUNT, batch_size):
        batch = requests[start:start + batch_size]
        outputs = generate_batch(batch)

        for j, candidate in enumerate(outputs):
            request_index = start + j
            reference = references[request_index]

            common = min(len(reference), len(candidate))
            mismatch = None

            for token_index in range(common):
                if reference[token_index].item() != candidate[token_index].item():
                    mismatch = token_index
                    break

            same = (
                mismatch is None
                and len(reference) == len(candidate)
            )

            exact_matches += int(same)
            total += 1

            if not same:
                print(
                    f"Request {request_index} "
                    f"({batch[j]['prompt_id']}): "
                    f"first mismatch = "
                    f"{mismatch + 1 if mismatch is not None else 'length difference'}, "
                    f"reference length = {len(reference)}, "
                    f"batch length = {len(candidate)}"
                )

    print(f"Exact matches: {exact_matches}/{total}")
    print(f"Match percentage: {100 * exact_matches / total:.2f}%")

del model
torch.cuda.empty_cache()
