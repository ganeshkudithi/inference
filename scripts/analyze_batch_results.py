#!/usr/bin/env python3

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def mean(values):
    return statistics.mean(values) if values else 0.0


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to write: {path}")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "batch_size",
        "run",
        "request_count",
        "generated_tokens_total",
        "batch_latency_seconds",
        "peak_allocated_gib",
        "peak_reserved_gib",
    }

    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    for row in rows:
        for key in required:
            row[key] = float(row[key])

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Analyze InferX static batching benchmark results."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to a benchmark_batch.py output CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results"),
        help="Root output directory (default: project results directory)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_root = Path(args.output_dir)
    processed_dir = output_root / "processed"
    figures_dir = output_root / "figures"

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)

    # One record per complete benchmark run and batch size.
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["batch_size"]), int(row["run"]))].append(row)

    run_summaries = []

    for (batch_size, run_id), batch_rows in sorted(groups.items()):
        total_tokens = sum(r["generated_tokens_total"] for r in batch_rows)
        total_seconds = sum(r["batch_latency_seconds"] for r in batch_rows)
        total_requests = sum(r["request_count"] for r in batch_rows)

        run_summaries.append({
            "source_csv": csv_path.name,
            "precision": batch_rows[0].get("precision", ""),
            "gpu": batch_rows[0].get("gpu", ""),
            "batch_size": batch_size,
            "run": run_id,
            "batches": len(batch_rows),
            "requests": int(total_requests),
            "generated_tokens": int(total_tokens),
            "generation_seconds": round(total_seconds, 6),
            "aggregate_tokens_per_second": round(
                total_tokens / total_seconds if total_seconds else 0.0, 6
            ),
            "mean_batch_latency_seconds": round(
                mean([r["batch_latency_seconds"] for r in batch_rows]), 6
            ),
            "max_peak_allocated_gib": round(
                max(r["peak_allocated_gib"] for r in batch_rows), 6
            ),
            "max_peak_reserved_gib": round(
                max(r["peak_reserved_gib"] for r in batch_rows), 6
            ),
        })

    run_path = processed_dir / "batch_run_summary.csv"
    write_csv(run_path, run_summaries)

    # Summarize independent runs, not individual batches.
    by_size = defaultdict(list)
    for row in run_summaries:
        by_size[int(row["batch_size"])].append(row)

    aggregate = []

    for batch_size in sorted(by_size):
        runs = by_size[batch_size]
        throughputs = [r["aggregate_tokens_per_second"] for r in runs]
        durations = [r["generation_seconds"] for r in runs]
        memory = [r["max_peak_allocated_gib"] for r in runs]
        reserved = [r["max_peak_reserved_gib"] for r in runs]

        aggregate.append({
            "source_csv": csv_path.name,
            "batch_size": batch_size,
            "runs": len(runs),
            "requests_per_run_min": min(r["requests"] for r in runs),
            "requests_per_run_max": max(r["requests"] for r in runs),
            "mean_generated_tokens_per_run": round(
                mean([r["generated_tokens"] for r in runs]), 3
            ),
            "mean_throughput_tokens_per_second": round(mean(throughputs), 4),
            "std_throughput_tokens_per_second": round(std(throughputs), 4),
            "mean_generation_seconds": round(mean(durations), 4),
            "std_generation_seconds": round(std(durations), 4),
            "mean_peak_allocated_gib": round(mean(memory), 4),
            "max_peak_allocated_gib": round(max(memory), 4),
            "mean_peak_reserved_gib": round(mean(reserved), 4),
        })

    baseline = next(
        (r["mean_throughput_tokens_per_second"]
         for r in aggregate if r["batch_size"] == 1),
        None,
    )

    for row in aggregate:
        row["speedup_vs_batch1"] = (
            round(row["mean_throughput_tokens_per_second"] / baseline, 4)
            if baseline else ""
        )

    aggregate_path = processed_dir / "batch_aggregate_summary.csv"
    write_csv(aggregate_path, aggregate)

    # Graphs show variability between complete runs.
    sizes = [r["batch_size"] for r in aggregate]
    throughput = [r["mean_throughput_tokens_per_second"] for r in aggregate]
    throughput_std = [r["std_throughput_tokens_per_second"] for r in aggregate]
    durations = [r["mean_generation_seconds"] for r in aggregate]
    duration_std = [r["std_generation_seconds"] for r in aggregate]

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        sizes, throughput, yerr=throughput_std,
        marker="o", capsize=5
    )
    plt.xticks(sizes)
    plt.xlabel("Static batch size")
    plt.ylabel("Aggregate throughput (tokens/s)")
    plt.title("InferX: Static Batching Throughput")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    throughput_path = figures_dir / "batch_throughput_summary.png"
    plt.savefig(throughput_path, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        sizes, durations, yerr=duration_std,
        marker="o", capsize=5
    )
    plt.xticks(sizes)
    plt.xlabel("Static batch size")
    plt.ylabel("Generation time per run (seconds)")
    plt.title("InferX: Static Batching Generation Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    latency_path = figures_dir / "batch_latency_summary.png"
    plt.savefig(latency_path, dpi=180)
    plt.close()

    print("\nAGGREGATE RESULTS")
    print("-" * 100)
    print(
        f"{'Batch':>5} {'Runs':>5} {'Tok/s mean':>12} "
        f"{'Tok/s std':>11} {'Time mean(s)':>14} "
        f"{'Speedup':>9} {'Peak GiB':>10}"
    )

    for r in aggregate:
        speedup = r["speedup_vs_batch1"]
        speedup_text = f"{speedup:.2f}x" if isinstance(speedup, float) else "N/A"
        print(
            f"{r['batch_size']:>5} "
            f"{r['runs']:>5} "
            f"{r['mean_throughput_tokens_per_second']:>12.2f} "
            f"{r['std_throughput_tokens_per_second']:>11.2f} "
            f"{r['mean_generation_seconds']:>14.2f} "
            f"{speedup_text:>9} "
            f"{r['max_peak_allocated_gib']:>10.2f}"
        )

    print("\nSaved:")
    print(f"  {run_path}")
    print(f"  {aggregate_path}")
    print(f"  {throughput_path}")
    print(f"  {latency_path}")

    if len({r["requests"] for r in run_summaries}) > 1:
        print(
            "\nWARNING: request counts differ across runs. "
            "Inspect the run summary before comparing performance."
        )

    token_counts = {
        r["batch_size"]: [
            x["generated_tokens"] for x in run_summaries
            if x["batch_size"] == r["batch_size"]
        ]
        for r in aggregate
    }
    token_means = {
        size: mean(values) for size, values in token_counts.items()
    }

    if len(token_means) > 1 and not math.isclose(
        max(token_means.values()), min(token_means.values()),
        rel_tol=0.01,
    ):
        print(
            "\nWARNING: mean generated-token totals vary by more than 1% "
            "between batch sizes. Investigate output-length differences "
            "before interpreting speedup."
        )


if __name__ == "__main__":
    main()
