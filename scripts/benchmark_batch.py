
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sync_gpu():
    torch.cuda.synchronize()


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def count_generated_tokens(sequence, input_length, eos_ids):
    """Count generated tokens, including the first EOS token if present."""
    generated = sequence[input_length:]
    if eos_ids:
        for index, token in enumerate(generated):
            if token.item() in eos_ids:
                return index + 1
    return len(generated)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark static batching for LLM inference."
    )
    parser.add_argument("--config", default="configs/bf16.json")
    parser.add_argument("--prompts", default="workloads/prompts.jsonl")
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8]
    )
    parser.add_argument(
        "--requests-per-run",
        type=int,
        default=24,
        help="Total requests processed per measured run; must be divisible "
             "by every batch size.",
    )
    args = parser.parse_args()

    config = load_json(args.config)
    model_path = Path(config["model_path"]).expanduser().resolve()
    precision = config.get("precision", "bf16").lower()
    max_new_tokens = int(config.get("max_new_tokens", 128))
    warmup_runs = int(config.get("warmup_runs", 2))
    benchmark_runs = int(config.get("benchmark_runs", 5))

    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run inside a GPU allocation.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this GPU.")
    if max_new_tokens < 1 or warmup_runs < 0 or benchmark_runs < 1:
        raise ValueError("Invalid generation or benchmark configuration.")
    if not args.batch_sizes or any(b < 1 for b in args.batch_sizes):
        raise ValueError("Batch sizes must be positive integers.")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("Batch sizes must not contain duplicates.")
    if args.requests_per_run < 1:
        raise ValueError("--requests-per-run must be positive.")
    for batch_size in args.batch_sizes:
        if args.requests_per_run % batch_size != 0:
            raise ValueError(
                f"{args.requests_per_run} requests is not divisible by "
                f"batch size {batch_size}. Choose a request count divisible "
                "by every requested batch size."
            )
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid local model directory: {model_path}")

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
    for item in prompts:
        if not item.get("id") or not isinstance(item.get("prompt"), str):
            raise ValueError("Each prompt needs an id and a prompt string.")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    device = torch.device("cuda:0")

    print("=" * 60)
    print("InferX Static Batch Benchmark")
    print("=" * 60)
    print("Model:", model_path)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Precision:", precision)
    print("Batch sizes:", args.batch_sizes)
    print("Requests per measured run:", args.requests_per_run)
    print("Warm-up runs:", warmup_runs)
    print("Measured runs:", benchmark_runs)
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    sync_gpu()
    model_load_seconds = time.perf_counter() - load_start

    # Deterministic generation for reproducible comparisons.
    model.generation_config.do_sample = False

    eos_value = model.generation_config.eos_token_id
    if eos_value is None:
        eos_value = tokenizer.eos_token_id
    if eos_value is None:
        eos_ids = set()
    elif isinstance(eos_value, int):
        eos_ids = {eos_value}
    else:
        eos_ids = set(eos_value)

    formatted_prompts = []
    for item in prompts:
        messages = [{"role": "user", "content": item["prompt"]}]
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = item["prompt"]
        formatted_prompts.append({"id": item["id"], "text": text})

    print(f"Model load time: {model_load_seconds:.3f} seconds")

    # Use the same ordered request workload for every batch size.
    # Prompt IDs repeat cyclically when the request count exceeds
    # the number of unique prompts.
    request_pool = [
        formatted_prompts[i % len(formatted_prompts)]
        for i in range(args.requests_per_run)
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = output_dir / f"batch_{precision}_{timestamp}.csv"
    rows = []

    for batch_size in args.batch_sizes:
        print(f"\n--- Batch size: {batch_size} ---")

        # Split the same number of requests into full batches.
        prepared_batches = []
        for start in range(0, len(request_pool), batch_size):
            batch = request_pool[start:start + batch_size]
            encoded = tokenizer(
                [item["text"] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            # Move inputs before timing generation.
            encoded = {
                key: value.to(device) for key, value in encoded.items()
            }
            prepared_batches.append((batch, encoded))

        def generate(encoded):
            with torch.inference_mode():
                return model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

        # Warm up a representative batch before collecting measurements.
        for _ in range(warmup_runs):
            generate(prepared_batches[0][1])
        sync_gpu()

        for run_idx in range(1, benchmark_runs + 1):
            run_elapsed = 0.0
            run_generated_tokens = 0

            for batch_idx, (batch, encoded) in enumerate(
                prepared_batches, start=1
            ):
                sync_gpu()
                torch.cuda.reset_peak_memory_stats(device)

                start_time = time.perf_counter()
                output = generate(encoded)
                sync_gpu()
                elapsed = time.perf_counter() - start_time

                input_length = encoded["input_ids"].shape[1]
                token_counts = [
                    count_generated_tokens(
                        output[i], input_length, eos_ids
                    )
                    for i in range(len(batch))
                ]
                generated_total = sum(token_counts)
                throughput = generated_total / elapsed if elapsed else 0.0

                peak_allocated = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                )
                peak_reserved = (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                )

                rows.append({
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "precision": precision,
                    "gpu": torch.cuda.get_device_name(0),
                    "batch_size": batch_size,
                    "run": run_idx,
                    "batch_index": batch_idx,
                    "request_count": len(batch),
                    "prompt_ids": ";".join(x["id"] for x in batch),
                    "input_tokens_padded_per_request": input_length,
                    "generated_tokens_total": generated_total,
                    "batch_latency_seconds": elapsed,
                    "throughput_tokens_per_second": throughput,
                    "amortized_batch_latency_per_request_seconds":
                        elapsed / len(batch),
                    "peak_allocated_gib": peak_allocated,
                    "peak_reserved_gib": peak_reserved,
                    "model_load_seconds": model_load_seconds,
                })

                run_elapsed += elapsed
                run_generated_tokens += generated_total

                print(
                    f"batch={batch_size} | run={run_idx}/{benchmark_runs} | "
                    f"batch={batch_idx}/{len(prepared_batches)} | "
                    f"latency={elapsed:.3f}s | "
                    f"generated={generated_total} tokens | "
                    f"throughput={throughput:.2f} tokens/s | "
                    f"memory={peak_allocated:.3f} GiB"
                )

            # Aggregate throughput across the entire identical request
            # workload for this run, rather than averaging batch rates.
            print(
                f"RUN SUMMARY | batch_size={batch_size} | run={run_idx} | "
                f"requests={args.requests_per_run} | "
                f"generation_time={run_elapsed:.3f}s | "
                f"generated_tokens={run_generated_tokens} | "
                f"aggregate_throughput="
                f"{run_generated_tokens / run_elapsed:.2f} tokens/s"
            )

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nBenchmark completed.")
    print("Batch executions recorded:", len(rows))
    print("Results saved to:", output_file)


if __name__ == "__main__":
    main()

