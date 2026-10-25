
import argparse
import csv
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


def sync_gpu():
    """Wait for pending GPU operations to finish."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_json(path):
    """Load and validate a JSON file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path}, line {exc.lineno}: {exc.msg}"
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM latency and throughput."
    )
    parser.add_argument(
        "--config",
        default="configs/bf16.json",
        help="Path to benchmark configuration JSON.",
    )
    parser.add_argument(
        "--prompts",
        default="workloads/prompts.jsonl",
        help="Path to JSONL prompt dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/raw",
        help="Directory for raw benchmark CSV files.",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Load and validate configuration
    # ---------------------------------------------------------
    config = load_json(args.config)

    required_keys = (
        "model_path",
        "max_new_tokens",
        "warmup_runs",
        "benchmark_runs",
    )

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing configuration key: {key}")

    max_new_tokens = int(config["max_new_tokens"])
    warmup_runs = int(config["warmup_runs"])
    benchmark_runs = int(config["benchmark_runs"])

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative.")

    if benchmark_runs < 2:
        raise ValueError(
            "benchmark_runs must be at least 2 to assess variability."
        )

    precision = config.get("precision", "bf16").lower()

    precision_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if precision not in precision_map:
        raise ValueError(
            f"Unsupported precision: {precision}. "
            f"Choose from {list(precision_map)}."
        )

    model_path = Path(config["model_path"]).expanduser().resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {model_path}"
        )

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Model config.json not found in {model_path}"
        )

    # ---------------------------------------------------------
    # 2. Load and validate prompts
    # ---------------------------------------------------------
    prompt_path = Path(args.prompts)

    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Prompt dataset not found: {prompt_path}"
        )

    prompts = []

    for line_number, line in enumerate(
        prompt_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {prompt_path}, line {line_number}"
            ) from exc

        if not item.get("id") or not isinstance(item.get("prompt"), str):
            raise ValueError(
                f"Each prompt needs a non-empty 'id' and a string "
                f"'prompt' (line {line_number})."
            )

        prompts.append(item)

    if not prompts:
        raise ValueError("No prompts found.")

    # ---------------------------------------------------------
    # 3. Validate GPU and precision
    # ---------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script in a GPU allocation."
        )

    if (
        precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("BF16 is not supported on this GPU.")

    dtype = precision_map[precision]
    device = torch.device("cuda:0")

    print("=" * 60)
    print("InferX Latency Benchmark")
    print("=" * 60)
    print("Model:", model_path)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Precision:", precision)
    print("Prompt count:", len(prompts))
    print("Warm-up runs per prompt:", warmup_runs)
    print("Measured runs per prompt:", benchmark_runs)
    print("Maximum new tokens:", max_new_tokens)
    print("Total measured requests:", len(prompts) * benchmark_runs)
    print("=" * 60)

    # ---------------------------------------------------------
    # 4. Load model and tokenizer once
    # ---------------------------------------------------------
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

    # Use deterministic greedy decoding.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    sync_gpu()
    model_load_seconds = time.perf_counter() - load_start

    print(f"Model load time: {model_load_seconds:.3f} seconds")

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    if pad_token_id is None:
        raise ValueError(
            "Tokenizer has neither a pad token nor an EOS token."
        )

    rows = []

    # ---------------------------------------------------------
    # 5. Benchmark each prompt
    # ---------------------------------------------------------
    for prompt_index, prompt in enumerate(prompts, start=1):
        print(
            f"\nPrompt {prompt_index}/{len(prompts)}: "
            f"{prompt['id']}"
        )

        messages = [
            {"role": "user", "content": prompt["prompt"]}
        ]

        if tokenizer.chat_template:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt["prompt"]

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(device)

        input_tokens = inputs["input_ids"].shape[1]

        # Warm-up requests are excluded from the CSV.
        for warmup_index in range(warmup_runs):
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                )

            sync_gpu()

        # -----------------------------------------------------
        # 6. Repeated measured trials
        # -----------------------------------------------------
        for run_index in range(1, benchmark_runs + 1):
            torch.cuda.reset_peak_memory_stats(device)

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=600,
            )

            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
                "pad_token_id": pad_token_id,
            }

            generation_state = {}

            def worker():
                try:
                    with torch.inference_mode():
                        generation_state["output"] = model.generate(
                            **generation_kwargs
                        )
                except Exception as exc:
                    generation_state["error"] = exc

            sync_gpu()
            start_time = time.perf_counter()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            first_chunk_time = None

            try:
                for chunk in streamer:
                    if first_chunk_time is None and chunk:
                        first_chunk_time = time.perf_counter()
            except Exception:
                # Do not leave a failed/incomplete trial in the results.
                thread.join()
                raise

            thread.join()

            sync_gpu()
            end_time = time.perf_counter()

            if "error" in generation_state:
                raise RuntimeError(
                    f"Generation failed for {prompt['id']}, "
                    f"run {run_index}"
                ) from generation_state["error"]

            if "output" not in generation_state:
                raise RuntimeError(
                    f"No generated output for {prompt['id']}, "
                    f"run {run_index}"
                )

            output = generation_state["output"]

            generated_tokens = (
                output.shape[1] - input_tokens
            )

            total_latency = end_time - start_time

            # This is first non-empty decoded text latency,
            # not exact first-token execution latency.
            ttft = (
                first_chunk_time - start_time
                if first_chunk_time is not None
                else float("nan")
            )

            # Approximation only: text streaming does not provide
            # exact timestamps for every generated token.
            decode_duration = (
                max(total_latency - ttft, 0.0)
                if ttft == ttft
                else float("nan")
            )

            if generated_tokens > 1 and decode_duration == decode_duration:
                approx_avg_itl = (
                    decode_duration / (generated_tokens - 1)
                )
            else:
                approx_avg_itl = float("nan")

            throughput = (
                generated_tokens / total_latency
                if total_latency > 0
                else 0.0
            )

            row = {
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "model_path": str(model_path),
                "precision": precision,
                "gpu": torch.cuda.get_device_name(0),
                "prompt_id": prompt["id"],
                "run": run_index,
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "model_load_seconds": model_load_seconds,
                "ttft_seconds": ttft,
                "end_to_end_seconds": total_latency,
                "approx_avg_itl_seconds": approx_avg_itl,
                "output_tokens_per_second": throughput,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device)
                    / 1024**3
                ),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device)
                    / 1024**3
                ),
            }

            rows.append(row)

            print(
                f"  Run {run_index}/{benchmark_runs} | "
                f"TTFT={ttft:.3f}s | "
                f"E2E={total_latency:.3f}s | "
                f"tokens={generated_tokens} | "
                f"throughput={throughput:.2f} tokens/s"
            )

    # ---------------------------------------------------------
    # 7. Save raw results
    # ---------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_file = output_dir / (
        f"latency_{precision}_{timestamp}.csv"
    )

    if not rows:
        raise RuntimeError("No benchmark measurements were recorded.")

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("Benchmark completed.")
    print("Measured requests:", len(rows))
    print("Results saved to:", output_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
