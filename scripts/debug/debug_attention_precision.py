import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "models/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True
)

inputs = tokenizer(
    "Explain how lithium-ion batteries work.",
    return_tensors="pt"
).to("cuda")


def run_test(dtype):
    print("\n" + "=" * 60)
    print("Testing:", dtype)
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=dtype,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="eager",
    )

    model.eval()

    layer = model.model.layers[0]
    attention = layer.self_attn

    def hook(name):
        def check(module, inp, output):
            if isinstance(output, torch.Tensor):
                x = output
            elif isinstance(output, (tuple, list)):
                x = next(
                    (v for v in output if isinstance(v, torch.Tensor)),
                    None
                )
            else:
                x = None

            if x is not None:
                print(
                    f"{name}: dtype={x.dtype}, "
                    f"finite={torch.isfinite(x).sum().item()}/{x.numel()}, "
                    f"NaN={torch.isnan(x).sum().item()}, "
                    f"max_abs={x.abs().max().item()}"
                )
        return check

    attention.q_proj.register_forward_hook(hook("Q"))
    attention.k_proj.register_forward_hook(hook("K"))
    attention.v_proj.register_forward_hook(hook("V"))

    layer.register_forward_hook(hook("Decoder layer"))

    with torch.inference_mode():
        output = model(**inputs)

    print(
        "Final logits finite:",
        torch.isfinite(output.logits).all().item()
    )

    del model
    torch.cuda.empty_cache()


run_test(torch.float16)
run_test(torch.bfloat16)
