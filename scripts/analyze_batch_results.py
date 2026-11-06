
#!/usr/bin/env python3

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def mean(values):
    return statistics.mean(values) if values else 0.0


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path, rows):
    """Write dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"WARNING: No data to write: {path}")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path):
    """Load and validate one benchmark CSV."""

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

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
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    numeric_fields = {
        "batch_size",
        "run",
        "request_count",
        "generated_tokens_total",
        "batch_latency_seconds",
        "peak_allocated_gib",
        "peak_reserved_gib",
    }

    for line_number, row in enumerate(rows, start=2):
        for key in numeric_fields:
            try:
                row[key] = float(row[key])
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid value in {path}, line {line_number}, "
                    f"column '{key}': {row.get(key)!r}"
                )

        if row["batch_size"] < 1 or not row["batch_size"].is_integer():
            raise ValueError(
                f"Invalid batch size in {path}, line {line_number}"
            )

        if row["run"] < 1 or not row["run"].is_integer():
            raise ValueError(
                f"Invalid run number in {path}, line {line_number}"
            )

        if row["request_count"] < 1:
            raise ValueError(
                f"Invalid request count in {path}, line {line_number}"
            )

        if row["generated_tokens_total"] < 0:
            raise ValueError(
                f"Negative generated-token count in {path}, "
                f"line {line_number}"
            )

        if (
            not math.isfinite(row["batch_latency_seconds"])
            or row["batch_latency_seconds"] <= 0
        ):
            raise ValueError(
                f"Invalid batch latency in {path}, line {line_number}"
            )

        for key in ("peak_allocated_gib", "peak_reserved_gib"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(
                    f"Invalid memory value in {path}, line {line_number}"
                )

    return rows


def configuration_label(row):
    """Create a readable label for a benchmark configuration."""

    return (
        f"{row['precision'].upper()} | "
        f"{row['attention'].upper()} | "
        f"TF32 {row['tf32'].upper()} | "
        f"{row['gpu']}"
    )


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Analyze multiple InferX static batching benchmark CSVs."
    )

    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more benchmark CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results"),
        help="Root output directory.",
    )

    args = parser.parse_args()

    # Create unique output names for every analysis execution.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output_root = Path(args.output_dir)
    processed_dir = output_root / "processed"
    figures_dir = output_root / "figures"

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Load multiple CSV files
    # -----------------------------------------------------------------

    all_rows = []
    input_files = []
    seen_files = set()

    for csv_file in args.csv:

        csv_path = Path(csv_file).expanduser().resolve()

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Benchmark CSV not found: {csv_path}"
            )

        # Prevent the same file from being counted twice.
        if csv_path in seen_files:
            print(f"Skipping duplicate input: {csv_path.name}")
            continue

        seen_files.add(csv_path)
        input_files.append(csv_path)

        file_rows = load_rows(csv_path)

        for row in file_rows:

            row["source_csv"] = csv_path.name

            # Preserve compatibility with older benchmark files.
            row["precision"] = row.get("precision") or "unknown"
            row["attention"] = row.get("attention") or "unknown"
            row["tf32"] = row.get("tf32") or "unknown"
            row["gpu"] = row.get("gpu") or "unknown"

            row["batch_size"] = int(row["batch_size"])
            row["run"] = int(row["run"])

        all_rows.extend(file_rows)

        print(
            f"Loaded {len(file_rows)} records from "
            f"{csv_path.name}"
        )

    rows = all_rows

    if not rows:
        raise ValueError("No benchmark records were loaded.")

    print("\nINPUT SUMMARY")
    print("-" * 70)
    print(f"Input files: {len(input_files)}")
    print(f"Total benchmark records: {len(rows)}")

    # -----------------------------------------------------------------
    # 2. Group individual batches into complete benchmark runs
    # -----------------------------------------------------------------

    # Source file is included here because run numbers restart at 1
    # in every benchmark CSV.
    run_groups = defaultdict(list)

    for row in rows:

        key = (
            row["source_csv"],
            row["precision"],
            row["attention"],
            row["tf32"],
            row["gpu"],
            row["batch_size"],
            row["run"],
        )

        run_groups[key].append(row)

    run_summaries = []

    for key, batch_rows in sorted(run_groups.items()):

        (
            source_csv,
            precision,
            attention,
            tf32,
            gpu,
            batch_size,
            run_id,
        ) = key

        total_tokens = sum(
            row["generated_tokens_total"]
            for row in batch_rows
        )

        total_seconds = sum(
            row["batch_latency_seconds"]
            for row in batch_rows
        )

        total_requests = sum(
            row["request_count"]
            for row in batch_rows
        )

        total_throughput = (
            total_tokens / total_seconds
            if total_seconds > 0
            else 0.0
        )

        run_summaries.append({

            "source_csv": source_csv,
            "precision": precision,
            "attention": attention,
            "tf32": tf32,
            "gpu": gpu,

            "batch_size": batch_size,
            "run": run_id,

            "batches": len(batch_rows),
            "requests": int(total_requests),
            "generated_tokens": int(total_tokens),

            "generation_seconds": round(total_seconds, 6),

            "aggregate_tokens_per_second": round(
                total_throughput, 6
            ),

            "mean_batch_latency_seconds": round(
                mean([
                    row["batch_latency_seconds"]
                    for row in batch_rows
                ]),
                6,
            ),

            "max_peak_allocated_gib": round(
                max([
                    row["peak_allocated_gib"]
                    for row in batch_rows
                ]),
                6,
            ),

            "max_peak_reserved_gib": round(
                max([
                    row["peak_reserved_gib"]
                    for row in batch_rows
                ]),
                6,
            ),
        })

    # Save run-level results.
    run_path = (
        processed_dir
        / f"batch_run_summary_{timestamp}.csv"
    )

    write_csv(run_path, run_summaries)

    print(f"\nComplete benchmark runs analyzed: {len(run_summaries)}")

    # -----------------------------------------------------------------
    # 3. Aggregate independent runs by configuration and batch size
    # -----------------------------------------------------------------

    # Multiple CSV files with the same configuration can contribute
    # independent measured runs to the same aggregate.
    aggregate_groups = defaultdict(list)

    for row in run_summaries:

        key = (
            row["precision"],
            row["attention"],
            row["tf32"],
            row["gpu"],
            row["batch_size"],
        )

        aggregate_groups[key].append(row)

    aggregate = []

    for key, runs in sorted(aggregate_groups.items()):

        (
            precision,
            attention,
            tf32,
            gpu,
            batch_size,
        ) = key

        throughputs = [
            row["aggregate_tokens_per_second"]
            for row in runs
        ]

        durations = [
            row["generation_seconds"]
            for row in runs
        ]

        memory = [
            row["max_peak_allocated_gib"]
            for row in runs
        ]

        reserved = [
            row["max_peak_reserved_gib"]
            for row in runs
        ]

        aggregate.append({

            "precision": precision,
            "attention": attention,
            "tf32": tf32,
            "gpu": gpu,

            "batch_size": batch_size,

            "runs": len(runs),

            "source_files": ";".join(sorted({
                row["source_csv"] for row in runs
            })),

            "requests_per_run_min": min(
                row["requests"] for row in runs
            ),

            "requests_per_run_max": max(
                row["requests"] for row in runs
            ),

            "mean_generated_tokens_per_run": round(
                mean([
                    row["generated_tokens"]
                    for row in runs
                ]),
                3,
            ),

            "mean_throughput_tokens_per_second": round(
                mean(throughputs), 4
            ),

            "std_throughput_tokens_per_second": round(
                std(throughputs), 4
            ),

            "mean_generation_seconds": round(
                mean(durations), 4
            ),

            "std_generation_seconds": round(
                std(durations), 4
            ),

            "mean_peak_allocated_gib": round(
                mean(memory), 4
            ),

            "max_peak_allocated_gib": round(
                max(memory), 4
            ),

            "mean_peak_reserved_gib": round(
                mean(reserved), 4
            ),
        })

    # -----------------------------------------------------------------
    # 4. Calculate speedup relative to batch size 1
    # -----------------------------------------------------------------

    baseline_by_configuration = {}

    for row in aggregate:

        if row["batch_size"] == 1:

            config_key = (
                row["precision"],
                row["attention"],
                row["tf32"],
                row["gpu"],
            )

            baseline_by_configuration[config_key] = (
                row["mean_throughput_tokens_per_second"]
            )

    for row in aggregate:

        config_key = (
            row["precision"],
            row["attention"],
            row["tf32"],
            row["gpu"],
        )

        baseline = baseline_by_configuration.get(config_key)

        if baseline is not None and baseline > 0:

            row["speedup_vs_batch1"] = round(
                row["mean_throughput_tokens_per_second"]
                / baseline,
                4,
            )

        else:
            row["speedup_vs_batch1"] = ""

    # Save aggregate results.
    aggregate_path = (
        processed_dir
        / f"batch_aggregate_summary_{timestamp}.csv"
    )

    write_csv(aggregate_path, aggregate)

    # -----------------------------------------------------------------
    # 5. Plot throughput and variability
    # -----------------------------------------------------------------

    configurations = defaultdict(list)

    for row in aggregate:

        config_key = (
            row["precision"],
            row["attention"],
            row["tf32"],
            row["gpu"],
        )

        configurations[config_key].append(row)

    plt.figure(figsize=(10, 6))

    for config_key, config_rows in sorted(configurations.items()):

        config_rows = sorted(
            config_rows,
            key=lambda row: row["batch_size"],
        )

        sizes = [
            row["batch_size"] for row in config_rows
        ]

        values = [
            row["mean_throughput_tokens_per_second"]
            for row in config_rows
        ]

        errors = [
            row["std_throughput_tokens_per_second"]
            for row in config_rows
        ]

        label = configuration_label({
            "precision": config_key[0],
            "attention": config_key[1],
            "tf32": config_key[2],
            "gpu": config_key[3],
        })

        plt.errorbar(
            sizes,
            values,
            yerr=errors,
            marker="o",
            capsize=4,
            label=label,
        )

    plt.xticks(sorted({
        row["batch_size"] for row in aggregate
    }))

    plt.xlabel("Static batch size")
    plt.ylabel("Aggregate throughput (tokens/s)")
    plt.title("InferX: Throughput Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    throughput_path = (
        figures_dir
        / f"batch_throughput_comparison_{timestamp}.png"
    )

    plt.savefig(throughput_path, dpi=180)
    plt.close()

    # -----------------------------------------------------------------
    # 6. Plot generation time
    # -----------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for config_key, config_rows in sorted(configurations.items()):

        config_rows = sorted(
            config_rows,
            key=lambda row: row["batch_size"],
        )

        sizes = [
            row["batch_size"] for row in config_rows
        ]

        values = [
            row["mean_generation_seconds"]
            for row in config_rows
        ]

        errors = [
            row["std_generation_seconds"]
            for row in config_rows
        ]

        label = configuration_label({
            "precision": config_key[0],
            "attention": config_key[1],
            "tf32": config_key[2],
            "gpu": config_key[3],
        })

        plt.errorbar(
            sizes,
            values,
            yerr=errors,
            marker="o",
            capsize=4,
            label=label,
        )

    plt.xticks(sorted({
        row["batch_size"] for row in aggregate
    }))

    plt.xlabel("Static batch size")
    plt.ylabel("Generation time per run (seconds)")
    plt.title("InferX: Generation Time Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    latency_path = (
        figures_dir
        / f"batch_latency_comparison_{timestamp}.png"
    )

    plt.savefig(latency_path, dpi=180)
    plt.close()

    # -----------------------------------------------------------------
    # 7. Print aggregate results
    # -----------------------------------------------------------------

    print("\nAGGREGATE RESULTS")
    print("=" * 120)

    print(
        f"{'Precision':<10}"
        f"{'Attention':<10}"
        f"{'TF32':<9}"
        f"{'Batch':>6}"
        f"{'Runs':>6}"
        f"{'Mean Tok/s':>13}"
        f"{'Std Tok/s':>12}"
        f"{'Mean Time':>12}"
        f"{'Speedup':>10}"
        f"{'Peak GiB':>11}"
    )

    print("-" * 120)

    for row in aggregate:

        speedup = row["speedup_vs_batch1"]

        speedup_text = (
            f"{speedup:.2f}x"
            if isinstance(speedup, (int, float))
            and speedup != ""
            else "N/A"
        )

        print(
            f"{row['precision']:<10}"
            f"{row['attention']:<10}"
            f"{row['tf32']:<9}"
            f"{row['batch_size']:>6}"
            f"{row['runs']:>6}"
            f"{row['mean_throughput_tokens_per_second']:>13.2f}"
            f"{row['std_throughput_tokens_per_second']:>12.2f}"
            f"{row['mean_generation_seconds']:>12.2f}"
            f"{speedup_text:>10}"
            f"{row['max_peak_allocated_gib']:>11.2f}"
        )

    # -----------------------------------------------------------------
    # 8. Data consistency checks
    # -----------------------------------------------------------------

    print("\nDATA VALIDATION")
    print("-" * 70)

    # Check request counts across batch sizes within each configuration
    request_groups = defaultdict(list)

    for row in run_summaries:

        config_key = (
            row["precision"],
            row["attention"],
            row["tf32"],
            row["gpu"],
            row["source_csv"],
        )

        request_groups[config_key].append(
            (row["batch_size"], row["requests"])
        )

    for config_key, values in request_groups.items():

        counts_by_size = defaultdict(list)

        for batch_size, requests in values:
            counts_by_size[batch_size].append(requests)

        all_counts = [
            count
            for counts in counts_by_size.values()
            for count in counts
        ]

        if len(set(all_counts)) > 1:

            print(
                "WARNING: Different request counts detected for "
                f"{config_key}. Check the run summary."
            )

    # Compare generated-token totals across batch sizes.
    token_groups = defaultdict(list)

    for row in run_summaries:

        config_key = (
            row["precision"],
            row["attention"],
            row["tf32"],
            row["gpu"],
            row["source_csv"],
            row["batch_size"],
        )

        token_groups[config_key].append(
            row["generated_tokens"]
        )

    token_means = defaultdict(dict)

    for key, values in token_groups.items():

        (
            precision,
            attention,
            tf32,
            gpu,
            source_csv,
            batch_size,
        ) = key

        config_key = (
            precision,
            attention,
            tf32,
            gpu,
            source_csv,
        )

        token_means[config_key][batch_size] = mean(values)

    for config_key, values in token_means.items():

        if len(values) > 1:

            maximum = max(values.values())
            minimum = min(values.values())

            if not math.isclose(
                maximum,
                minimum,
                rel_tol=0.01,
            ):

                print(
                    "WARNING: Mean generated-token totals differ by "
                    f"more than 1% across batch sizes for {config_key}."
                )

                print(
                    "         Investigate output-length differences "
                    "before interpreting throughput speedup."
                )

    if not token_means:
        print("No token consistency comparisons available.")

    # -----------------------------------------------------------------
    # 9. Print output locations
    # -----------------------------------------------------------------

    print("\nANALYSIS COMPLETED")
    print("=" * 70)

    print("\nInput files:")

    for path in input_files:
        print(f"  {path}")

    print("\nSaved outputs:")

    print(f"  Run summary:       {run_path}")
    print(f"  Aggregate summary: {aggregate_path}")
    print(f"  Throughput plot:   {throughput_path}")
    print(f"  Latency plot:      {latency_path}")

    print("\nOriginal raw benchmark CSV files were not modified.")


if __name__ == "__main__":
    main()
