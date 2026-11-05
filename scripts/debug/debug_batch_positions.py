import torch
from transformers import AutoTokenizer

MODEL = "models/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

tokenizer.padding_side = "left"

prompts = [
    "Explain what an electric vehicle is.",
    "Explain how lithium-ion batteries work in detail.",
]

encoded = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
)

attention_mask = encoded["attention_mask"]

position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)

print("Input IDs shape:", encoded["input_ids"].shape)

print("\nAttention mask:")
print(attention_mask)

print("\nCalculated position IDs:")
print(position_ids)

print("\nIndividual prompt position IDs:")
for i in range(len(prompts)):
    valid_positions = position_ids[i][attention_mask[i] == 1]
    print(f"Prompt {i}: {valid_positions.tolist()}")
