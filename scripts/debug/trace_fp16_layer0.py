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

layer = model.model.layers[0]

modules = {
    "input_layernorm": layer.input_layernorm,
    "q_proj": layer.self_attn.q_proj,
    "k_proj": layer.self_attn.k_proj,
    "v_proj": layer.self_attn.v_proj,
    "attention": layer.self_attn,
    "o_proj": layer.self_attn.o_proj,
    "post_attention_layernorm": layer.post_attention_layernorm,
    "mlp_gate_proj": layer.mlp.gate_proj,
    "mlp_up_proj": layer.mlp.up_proj,
    "mlp_down_proj": layer.mlp.down_proj,
    "mlp": layer.mlp,
}

def get_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    return None

def make_hook(name):
    def hook(module, inputs, output):
        tensor = get_tensor(output)

        if tensor is None:
            print(f"{name}: no tensor output")
            return

        finite = torch.isfinite(tensor)

        print(
            f"{name}: "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"finite={finite.sum().item()}/{tensor.numel()}, "
            f"NaN={torch.isnan(tensor).sum().item()}, "
            f"Inf={torch.isinf(tensor).sum().item()}"
        )

    return hook

for name, module in modules.items():
    module.register_forward_hook(make_hook(name))

inputs = tokenizer(
    "Explain how lithium-ion batteries work.",
    return_tensors="pt"
).to("cuda")

with torch.inference_mode():
    outputs = model(**inputs)

print("\nFinal logits finite:",
      torch.isfinite(outputs.logits).all().item())
