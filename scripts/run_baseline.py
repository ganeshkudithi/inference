
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--prompts", default="workloads/prompts.jsonl")
    parser.add_argument("--output-dir", default="results/raw")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    prompt_path = Path(args.prompts)

    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompts = [
        json.loads(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not prompts:
        raise ValueError("No prompts found.")

    for prompt in prompts:
        if "id" not in prompt or "prompt" not in prompt:
            raise ValueError("Each prompt needs 'id' and 'prompt' fields.")

    for key in ("model_path", "max_new_tokens", "warmup_runs", "benchmark_runs"):
        if key not in config:
            raise ValueError(f"Missing configuration key: {key}")

    if config["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    if config["warmup_runs"] < 0 or config["benchmark_runs"] < 1:
        raise ValueError("Invalid warmup_runs or benchmark_runs.")

    model_path = Path(config["model_path"]).expanduser().resolve()
    revision = config.get("revision")

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {model_path}"
        )

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"config.json not found in model directory: {model_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this benchmark in a GPU allocation."
        )

    device = torch.device("cuda:0")
    precision = config.get("precision", "bf16").lower()

    precision_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if precision not in precision_map:
        raise ValueError(
            f"Unsupported precision: {precision}. "
            f"Choose from {list(precision_map)}"
        )

    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this GPU.")

    dtype = precision_map[precision]

    print(f"Model path: {model_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Precision: {dtype}")
    print(f"Benchmark prompts: {len(prompts)}")

    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    # This benchmark uses greedy decoding, not sampling.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    sync_gpu()
    load_seconds = time.perf_counter() - load_start
    print(f"Model load time: {load_seconds:.2f} seconds")

    def generate(prompt):
        messages = [{"role": "user", "content": prompt}]

        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(device)

        input_tokens = inputs["input_ids"].shape[1]

        torch.cuda.reset_peak_memory_stats(device)
        sync_gpu()
        start = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )

        sync_gpu()
        elapsed = time.perf_counter() - start

        generated_tokens = output.shape[1] - input_tokens

        return {
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "latency_seconds": elapsed,
            "output_tokens_per_second": (
                generated_tokens / elapsed if elapsed > 0 else 0.0
            ),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / 1024**3
            ),
        }

    print("Running warm-up...")
    for _ in range(config["warmup_runs"]):
        generate(prompts[0]["prompt"])

    rows = []

    for prompt in prompts:
        for run in range(config["benchmark_runs"]):
            metrics = generate(prompt["prompt"])

            row = {
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "model_path": str(model_path),
                "revision": revision or "local",
                "gpu": torch.cuda.get_device_name(0),
                "dtype": str(dtype),
                "prompt_id": prompt["id"],
                "run": run + 1,
                "load_seconds": load_seconds,
                **metrics,
            }
            rows.append(row)

            print(
                f"{prompt['id']} | run {run + 1} | "
                f"latency={metrics['latency_seconds']:.3f}s | "
                f"throughput="
                f"{metrics['output_tokens_per_second']:.2f} tokens/s"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"{precision}_{timestamp}.csv"

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

