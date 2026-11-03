# InferX

**InferX** is an experimental framework for benchmarking and evaluating
Large Language Model (LLM) inference optimizations on GPU systems.

The project currently uses **Qwen2.5-1.5B-Instruct** on an NVIDIA A100
80GB GPU to study inference performance under different numerical
precisions and static batch sizes. The objective is to measure
throughput, generation latency, and GPU memory consumption while
maintaining reliable output behavior.

> **Project status:** Baseline inference and initial static-batching
> benchmarks have been implemented. Further validation and optimization
> experiments are in progress.

## Objectives

-   Establish a reproducible LLM inference baseline.
-   Compare numerical precision modes such as BF16, FP16, and FP32.
-   Evaluate static batching at different batch sizes.
-   Measure generation throughput, latency, and GPU memory usage.
-   Validate generated token sequences against a batch-size-1 reference.
-   Extend the benchmark framework to study additional optimizations,
    including quantization and KV-cache-related techniques.

## Current environment

  Component               Configuration
  ----------------------- ---------------------------
  Model                   Qwen2.5-1.5B-Instruct
  GPU                     NVIDIA A100 80GB PCIe
  CUDA                    12.4
  PyTorch                 2.6.0+cu124
  Transformers            4.57.1
  Python                  3.11
  Execution environment   Slurm-managed GPU cluster

The model is loaded from a local directory. The current cluster workflow
is designed for offline execution.

## Repository structure

``` text
InferX/
├── configs/
│   ├── baseline.json
│   ├── bf16.json
│   └── fp16.json
├── models/
│   └── Qwen2.5-1.5B-Instruct/   # Local model files; not committed
├── scripts/
│   ├── run_baseline.py
│   ├── benchmark_latency.py
│   ├── benchmark_batch.py
│   ├── validate_batch_correctness.py
│   ├── analyze_batch_results.py
│   ├── inspect_environment.py
│   └── test_cuda.py
├── workloads/
│   └── prompts.jsonl
├── results/
│   ├── raw/                     # Raw benchmark CSVs; not committed
│   ├── processed/               # Aggregated results; not committed
│   └── figures/                 # Generated plots; not committed
├── src/
├── tests/
├── docs/
├── README.md
└── requirements-lock.txt
```

## Setup

### 1. Clone the repository

``` bash
git clone https://github.com/ganeshkudithi/inference.git
cd inference
```

### 2. Activate the environment

Activate the Conda environment configured for the project:

``` bash
conda activate inferx
```

Install dependencies from the project's dependency file if required:

``` bash
pip install -r requirements-lock.txt
```

### 3. Configure the local model path

Download or copy the model into:

``` text
models/Qwen2.5-1.5B-Instruct/
```

The model directory is intentionally excluded from Git because model
weights are large. Update the `model_path` field in the JSON
configuration files if your local path differs.

For offline execution:

``` bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Running on a Slurm GPU cluster

Run GPU workloads only inside a GPU allocation provided by your cluster.
Do not launch GPU benchmarks on a login node.

An example interactive allocation is:

``` bash
salloc --partition=<gpu_partition> --gres=gpu:1 --time=01:00:00
```

Partition and resource options vary by cluster; use the options
specified by your system administrator.

After the allocation starts, verify the assigned GPU:

``` bash
echo "$SLURM_JOB_ID"
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi
```

Activate the environment and move to the project directory before
running experiments.

## Benchmark methodology

### Numerical precision

The framework supports precision configurations through JSON files. The
current experiments include BF16 and FP32, with FP16 also available in
the configuration workflow.

Precision comparisons should use the same model, prompts, generation
settings, request count, and measurement procedure. Record the GPU and
software environment with every run.

### Static batching

Static batching groups a fixed number of independent prompts into one
model-generation call.

For example, with 24 requests:

    Batch size   Number of generation calls
  ------------ ----------------------------
             1                           24
             2                           12
             4                            6
             8                            3

A larger batch can improve GPU utilization and aggregate throughput. It
can also increase memory use and may affect individual request latency.
Static batching in this project is an offline fixed-workload experiment;
it is not a measurement of a production online-serving scheduler.

### Metrics

The benchmark scripts record or derive metrics including:

-   **Throughput:** generated tokens divided by generation time.
-   **Batch latency:** elapsed time for a batch-generation call.
-   **Amortized latency per request:** batch latency divided by the
    number of requests in that batch.
-   **Peak allocated GPU memory:** maximum memory allocated by PyTorch
    during the measured run.
-   **Peak reserved GPU memory:** maximum memory reserved by PyTorch.
-   **Correctness:** generated token sequences compared with the
    batch-size-1 reference.

Timing boundaries matter. The static-batching benchmark measures
generation after tokenization and input transfer; model loading and
those preparation steps are not included in generation time. Do not
compare these timings directly with end-to-end serving latency.

## Running experiments

### Baseline inference

``` bash
python scripts/run_baseline.py --config configs/bf16.json
```

### Latency benchmark

``` bash
python scripts/benchmark_latency.py --config configs/bf16.json
```

### Static-batching benchmark

The following example tests batch sizes 1, 2, 4, and 8 using 24 requests
per run:

``` bash
python scripts/benchmark_batch.py \
  --config configs/bf16.json \
  --batch-sizes 1 2 4 8 \
  --requests-per-run 24
```

For an FP32 experiment, use the FP32 configuration file if one is
available in your working copy, or create a configuration with
`"precision": "fp32"` and the correct local model path.

### Correctness validation

``` bash
python scripts/validate_batch_correctness.py \
  --config configs/bf16.json \
  --batch-sizes 1 2 4 8 \
  --requests-per-run 24
```

The validator compares each request's generated token IDs against its
batch-size-1 reference. A throughput improvement should not be treated
as a successful optimization until output behavior has also been
checked.

### Analyze benchmark results

``` bash
python scripts/analyze_batch_results.py
```

Check the script's command-line help if your local version requires an
input CSV or additional arguments:

``` bash
python scripts/analyze_batch_results.py --help
```

## Experimental results

### FP32 static-batching benchmark

The following results are from a five-run FP32 static-batching
experiment on an NVIDIA A100 80GB PCIe GPU. Each run processed 24
requests. The workload generated 2,448 tokens in total per run.

  -----------------------------------------------------------------------
         Batch size         Aggregate   Generation time Speedup vs. batch
                           throughput               (s)                 1
                           (tokens/s)                   
  ----------------- ----------------- ----------------- -----------------
                  1             49.66             49.30             1.00×

                  2             71.40             34.29             1.44×

                  4            141.17             17.34             2.84×

                  8            272.46              8.99             5.49×
  -----------------------------------------------------------------------

Throughput is calculated as total generated tokens divided by total
generation time. The speedup is calculated relative to batch size 1.

#### GPU memory

    Batch size   Peak allocated (GiB)   Peak reserved (GiB)
  ------------ ---------------------- ---------------------
             1                   5.78                  5.93
             2                   5.80                  5.95
             4                   5.83                  5.98
             8                   5.89                  6.03

#### Observations

-   Aggregate throughput increased with batch size in this workload.
-   Batch size 8 achieved approximately 5.49× the throughput of batch
    size 1.
-   Peak allocated memory increased modestly from approximately 5.78 GiB
    to 5.89 GiB between batch sizes 1 and 8.
-   These measurements describe this model, GPU, workload, and benchmark
    setup; they should not be generalized to other models or hardware
    without further experiments.

**Correctness note:** The performance CSV establishes timing, token
counts, and memory measurements. It does not by itself establish exact
token-level equivalence between FP32 and BF16, or correctness for every
batch size. Consult the corresponding correctness-validation output
before making such claims.

### Preliminary BF16 comparison

Earlier BF16 measurements showed approximately 54.52, 68.26, 134.47, and
278.92 tokens/s for batch sizes 1, 2, 4, and 8, respectively. The
earlier BF16 runs did not always generate the same total number of
tokens across batch sizes. Therefore, treat direct precision comparisons
at batch sizes 2 and 4 as preliminary until the workload and output
lengths are controlled and correctness is revalidated.

## Reproducibility notes

For each experiment, retain:

1.  The exact configuration file.
2.  The raw per-run CSV output.
3.  The correctness-validation summary and details.
4.  The model identifier and local model revision, where available.
5.  GPU model, driver, CUDA, PyTorch, and Transformers versions.
6.  Workload prompts and generation parameters.

Use consistent prompts, request ordering, generation limits, EOS and
padding configuration, and measurement boundaries when comparing
experiments.

## Current limitations

-   The current workload contains a small set of prompts and should be
    expanded for broader evaluation.
-   Static batching does not model dynamic request arrivals or
    continuous batching.
-   Correctness differences observed in earlier BF16 batch tests require
    further investigation.
-   The current benchmark is not a full production-serving evaluation.
-   Quantization and additional inference optimizations remain future
    work.

## Planned work

-   Expand the prompt dataset and workload diversity.
-   Complete precision and batch-size correctness validation.
-   Investigate padding, position-ID, and batch-composition effects on
    generated outputs.
-   Add quantization experiments and compare memory, throughput, and
    output quality.
-   Study KV-cache behavior and memory usage.
-   Explore continuous batching and request-level latency metrics.
-   Consolidate results into reproducible tables and plots.

## Contributing

Contributions and suggestions are welcome. For reproducible results,
include the configuration, workload, environment details, raw
measurements, and correctness results with any benchmark report.

## License

Add the license and usage terms applicable to this repository and any
included code before public distribution.
