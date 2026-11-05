import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "models/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map={"": 0},
    local_files_only=True,
    attn_implementation="eager",
)

model.eval()

# Check whether model parameters already contain invalid values.
bad_parameters = []

for name, param in model.named_parameters():
    if not torch.isfinite(param).all():
        bad_parameters.append(name)

print("Parameters containing NaN/Inf:", bad_parameters[:10])
print("Total affected parameters:", len(bad_parameters))

# Register hooks to inspect activations.
modules = {
    "embedding": model.model.embed_tokens,
}

for i, layer in enumerate(model.model.layers):
    modules[f"decoder_layer_{i}"] = layer

modules["final_norm"] = model.model.norm
modules["lm_head"] = model.lm_head

first_bad = None

def check_output(name):
    def hook(module, inputs, output):
        global first_bad

        tensor = output[0] if isinstance(output, tuple) else output

        if not isinstance(tensor, torch.Tensor):
            return

        finite = torch.isfinite(tensor)

        if not finite.all():
            if first_bad is None:
                first_bad = name

            print(
                f"NON-FINITE at {name}: "
                f"NaN={torch.isnan(tensor).sum().item()}, "
                f"Inf={torch.isinf(tensor).sum().item()}"
            )
        else:
            print(
                f"{name}: "
                f"max_abs={tensor.abs().max().item():.5f}"
            )

    return hook

for name, module in modules.items():
    module.register_forward_hook(check_output(name))

inputs = tokenizer(
    "Explain how lithium-ion batteries work.",
    return_tensors="pt"
).to("cuda")

with torch.inference_mode():
    outputs = model(**inputs)

print("\nFirst component producing NaNs:", first_bad)
print("Final logits finite:", torch.isfinite(outputs.logits).all().item())
