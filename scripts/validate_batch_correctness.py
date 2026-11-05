import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="Validate greedy generation consistency across batch sizes."
    )
    parser.add_argument("--config", default="configs/bf16.json")
    parser.add_argument("--prompts", default="workloads/prompts.jsonl")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument(
        "--attention",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="Attention implementation to use.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run inside a GPU allocation.")

    if args.requests < 1 or any(b < 1 for b in args.batch_sizes):
        raise ValueError("Request count and batch sizes must be positive.")

    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("Batch sizes must be unique.")

    for size in args.batch_sizes:
        if args.requests % size:
            raise ValueError(
                f"Request count {args.requests} must be divisible by {size}."
            )

    config = load_json(args.config)
    model_path = Path(config["model_path"]).expanduser().resolve()
    precision = config.get("precision", "bf16").lower()
    max_new_tokens = int(config.get("max_new_tokens", 128))

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if precision not in dtypes:
        raise ValueError(f"Unsupported precision: {precision}")

    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the current GPU.")

    prompts = [
        json.loads(line)
        for line in Path(args.prompts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError("No prompts found.")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Match the existing benchmark's padding convention.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtypes,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation=args.attention,
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     str(model_path),
    #     dtype=dtypes[precision],
    #     device_map={"": 0},
    #     local_files_only=True,
    #     trust_remote_code=False,
    # )
    model.eval()

    # eos_value = model.generation_config.eos_token_id
    # if eos_value is None:
    #     eos_value = tokenizer.eos_token_id

    # if eos_value is None:
    #     eos_ids = set()
    # elif isinstance(eos_value, int):
    #     eos_ids = {eos_value}
    # else:
    #     eos_ids = set(eos_value)
    eos_ids = {tokenizer.eos_token_id}

    formatted = []
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

        formatted.append({"id": item["id"], "text": text})

    # Same ordered workload for every batch size.
    requests = [
        {
            "request_index": i,
            "prompt_id": formatted[i % len(formatted)]["id"],
            "text": formatted[i % len(formatted)]["text"],
        }
        for i in range(args.requests)
    ]

    output_records = {}
    device = torch.device("cuda:0")

    print("Model:", model_path)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Precision:", precision)
    print("Requests:", len(requests))
    print("Batch sizes:", args.batch_sizes)
    print("Max new tokens:", max_new_tokens)
    print("Attention:", args.attention)

    for batch_size in args.batch_sizes:
        print(f"\nValidating batch size {batch_size}")
        results = {}

        for start in range(0, len(requests), batch_size):
            batch = requests[start:start + batch_size]

            encoded = tokenizer(
                [item["text"] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            padded_input_length = encoded["input_ids"].shape[1]

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            for i, item in enumerate(batch):
                # Decoder-only generation appends new tokens after the
                # complete padded input sequence.
                generated = output[i, padded_input_length:].tolist()

                # Include the first EOS token; discard subsequent padding.
                if eos_ids:
                    for j, token in enumerate(generated):
                        if token in eos_ids:
                            generated = generated[:j + 1]
                            break

                results[item["request_index"]] = {
                    "request_index": item["request_index"],
                    "prompt_id": item["prompt_id"],
                    "token_ids": generated,
                    "generated_tokens": len(generated),
                    "ended_with_eos": bool(generated and generated[-1] in eos_ids),
                    "batch_latency_seconds": round(elapsed, 6),
                }

        output_records[batch_size] = results

        lengths = [r["generated_tokens"] for r in results.values()]
        print(
            f"  requests={len(results)}, "
            f"min_tokens={min(lengths)}, "
            f"max_tokens={max(lengths)}, "
            f"mean_tokens={sum(lengths) / len(lengths):.2f}"
        )

    # Batch size 1 is the reference for exact token-ID comparison.
    reference_size = 1 if 1 in output_records else args.batch_sizes[0]
    reference = output_records[reference_size]

    summary_rows = []
    detail_rows = []

    for batch_size in args.batch_sizes:
        exact_matches = 0
        total = len(requests)

        for request_index in range(total):
            ref = reference[request_index]
            candidate = output_records[batch_size][request_index]
            exact = ref["token_ids"] == candidate["token_ids"]
            exact_matches += int(exact)

            detail_rows.append({
                "batch_size": batch_size,
                "reference_batch_size": reference_size,
                "request_index": request_index,
                "prompt_id": candidate["prompt_id"],
                "reference_tokens": ref["generated_tokens"],
                "candidate_tokens": candidate["generated_tokens"],
                "exact_token_match": exact,
                "reference_eos": ref["ended_with_eos"],
                "candidate_eos": candidate["ended_with_eos"],
                "reference_token_ids": json.dumps(ref["token_ids"]),
                "candidate_token_ids": json.dumps(candidate["token_ids"]),
            })

        summary_rows.append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "precision": precision,
            "gpu": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
            "reference_batch_size": reference_size,
            "requests": total,
            "exact_matches": exact_matches,
            "exact_match_rate": exact_matches / total,
            "mean_generated_tokens": sum(
                r["generated_tokens"]
                for r in output_records[batch_size].values()
            ) / total,
            "eos_terminated_requests": sum(
                r["ended_with_eos"]
                for r in output_records[batch_size].values()
            ),
        })

        print(
            f"  exact matches vs batch {reference_size}: "
            f"{exact_matches}/{total} ({100 * exact_matches / total:.1f}%)"
        )

    output_dir = Path("results/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = output_dir / f"correctness_summary_{timestamp}.csv"
    details_path = output_dir / f"correctness_details_{timestamp}.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with details_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)

    print("\nSaved:")
    print(summary_path)
    print(details_path)
    print(
        "\nNote: exact greedy token matching is a strict diagnostic. "
        "A mismatch can result from small numerical differences during "
        "batched computation; investigate mismatches before concluding "
        "that batching is incorrect."
    )


if __name__ == "__main__":
    main()
