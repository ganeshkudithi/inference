# InferX — LLM Inference Optimization and Benchmarking

## 1. Project Overview

InferX is a benchmarking framework for evaluating and optimizing Large Language Model (LLM) inference performance. The project focuses on measuring inference latency, time to first token (TTFT), output throughput, and GPU memory consumption under different model configurations and workloads.

The initial implementation uses Hugging Face Transformers and a locally stored Qwen2.5-1.5B-Instruct model. The framework is being developed incrementally to support precision comparisons, workload analysis, batching experiments, and optimized inference engines.

### Objectives

- Establish a reproducible baseline for LLM inference.
- Compare BF16 and FP16 inference performance.
- Measure TTFT, end-to-end latency, output throughput, and GPU memory usage.
- Investigate the impact of input and output token lengths.
- Evaluate batching and concurrent-request workloads.
- Explore quantization and optimized inference engines.
- Generate reproducible benchmark results for analysis and reporting.

## 2. Hardware and Software Environment

### Hardware

| Component | Configuration |
|---|---|
| GPU | NVIDIA A100 80GB PCIe |
| GPU memory capacity | 80 GB |
| GPU driver | 550.54.15 |
| CUDA reported by NVIDIA-SMI | 12.4 |
| Cluster scheduler | Slurm |

### Software

| Component | Version |
|---|---|
| Operating system | AlmaLinux 8.9 |
| Python | 3.11.17 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.1 |
| Environment manager | Conda |

The benchmark uses the existing PyTorch CUDA installation. A separate CUDA compiler (`nvcc`) is not required for the current Transformers-based implementation.

## 3. Model and Workloads

### Model

- Model: Qwen2.5-1.5B-Instruct
- Hugging Face identifier: `Qwen/Qwen2.5-1.5B-Instruct`
- Model location: `models/Qwen2.5-1.5B-Instruct`
- Loading mode: Local files only
- Decoding: Greedy decoding
- KV cache: Enabled

The model is stored locally to support execution on the offline compute cluster.

### Workload Dataset

The initial workload dataset is stored in:

`workloads/prompts.jsonl`

Each JSONL record contains a prompt identifier and its corresponding text.

The initial workloads are:

| Workload | Input tokens | Generated tokens |
|---|---:|---:|
| Short | 40 | 50 |
| Medium | 57 | 128 |
| Long | 84 | 128 |

The configured maximum output length is 128 tokens. The short workload terminates after generating 50 tokens, while the medium and long workloads reach the configured limit.

Token counts refer to the model's formatted input, including any chat-template tokens.

## 4. Project Structure

```text
InferX/
├── configs/
│   ├── baseline.json
│   ├── bf16.json
│   └── fp16.json
├── models/
│   └── Qwen2.5-1.5B-Instruct/
├── src/
├── workloads/
│   └── prompts.jsonl
├── scripts/
│   ├── run_baseline.py
│   └── benchmark_latency.py
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
├── tests/
├── docs/
└── README.md
```

Additional scripts and configuration files may be added as the project evolves.

## 5. Environment Setup

Activate the Conda environment:

```bash
conda activate inferx
```

Navigate to the project directory:

```bash
cd /home/cdacapp01/wk-ganesh/multi-modal/inference/InferX
```

Enable offline model loading:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Verify the environment:

```bash
python -c "import torch, transformers; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Check GPU availability:

```bash
nvidia-smi
```

Run GPU benchmarks only through an authorized GPU allocation on the Slurm cluster.

## 6. Benchmark Configuration

The benchmark configuration is stored in JSON files.

Example: `configs/bf16.json`

```json
{
  "model_path": "/home/cdacapp01/wk-ganesh/multi-modal/inference/InferX/models/Qwen2.5-1.5B-Instruct",
  "revision": null,
  "precision": "bf16",
  "max_new_tokens": 128,
  "warmup_runs": 2,
  "benchmark_runs": 5,
  "seed": 42
}
```

Example: `configs/fp16.json`

The FP16 configuration uses the same model, prompts, maximum output length, warm-up count, and benchmark-run count, with `"precision": "fp16"`.

Configuration parameters:

| Parameter | Description |
|---|---|
| `model_path` | Path to the local model directory |
| `revision` | Optional model revision metadata |
| `precision` | Inference precision: BF16, FP16, or FP32 |
| `max_new_tokens` | Maximum number of generated tokens |
| `warmup_runs` | Number of warm-up generations per prompt |
| `benchmark_runs` | Number of measured trials per prompt |
| `seed` | Reserved configuration field; not currently used to control the benchmark |

The latency benchmark loads the model once, performs the configured warm-ups for each prompt, and then executes the measured trials.

## 7. Running the Benchmarks

### 7.1 Baseline Benchmark

Validate the script:

```bash
python -m py_compile scripts/run_baseline.py
```

Run the baseline:

```bash
python scripts/run_baseline.py --config configs/bf16.json
```

The baseline benchmark records end-to-end generation latency, output throughput, and GPU memory statistics.

### 7.2 Latency Benchmark

Validate the latency benchmark:

```bash
python -m py_compile scripts/benchmark_latency.py
```

Run BF16:

```bash
python scripts/benchmark_latency.py --config configs/bf16.json
```

Run FP16:

```bash
python scripts/benchmark_latency.py --config configs/fp16.json
```

With three prompts and five measured runs per prompt, each latency benchmark produces 15 measured records.

### 7.3 Output Files

Raw CSV results are saved in:

`results/raw/`

Example result files:

```text
results/raw/latency_bf16_20261025T185359Z.csv
results/raw/latency_fp16_20261025T185540Z.csv
```

Each execution creates a timestamped result file. Existing raw results should be preserved for reproducibility.

## 8. Performance Metrics

| Metric | Definition |
|---|---|
| TTFT | Time from starting generation until the first non-empty decoded text chunk is received |
| End-to-end latency | Total duration of the measured generation request |
| Output throughput | Number of generated tokens divided by end-to-end latency |
| Peak allocated memory | Maximum GPU memory allocated by PyTorch during the measured generation |
| Peak reserved memory | Maximum GPU memory reserved by PyTorch's caching allocator |
| Mean latency | Arithmetic mean of measured request latencies |
| Standard deviation | Measure of variation across repeated trials |

**Measurement limitations**

- TTFT currently measures the first non-empty streamed text chunk, not the exact execution time of the first generated token.
- Average inter-token latency is an approximation and is not based on individual token timestamps.
- Throughput is calculated from generated tokens divided by end-to-end latency. It is not pure decode-only throughput.
- The current benchmark executes requests sequentially. It does not measure concurrent serving or continuous batching.
- Peak memory values represent PyTorch allocator measurements, not total GPU memory usage by every process.
- Results depend on GPU utilization, clock behavior, workload characteristics, software versions, and measurement overhead.

## 9. Experimental Results

### 9.1 BF16 Baseline

Hardware: NVIDIA A100 80GB PCIe

Model: Qwen2.5-1.5B-Instruct

| Workload | Mean TTFT | Mean latency | Mean throughput | Peak allocated memory |
|---|---:|---:|---:|---:|
| Short | 40.2 ms | 0.9423 s | 53.0632 tokens/s | 2.8958 GiB |
| Medium | 39.8 ms | 2.4375 s | 52.5239 tokens/s | 2.8983 GiB |
| Long | 39.8 ms | 2.4424 s | 52.4130 tokens/s | 2.8990 GiB |

Five measured trials were collected for each workload, after two warm-up generations per prompt.

### 9.2 BF16 vs FP16

Overall averages across the three workloads:

| Metric | BF16 | FP16 |
|---|---:|---:|
| Mean TTFT | 39.9 ms | 40.1 ms |
| Mean end-to-end latency | 1.9407 s | 1.9362 s |
| Mean throughput | 52.6667 tokens/s | 52.8380 tokens/s |
| Peak allocated GPU memory | 2.899 GiB | 2.899 GiB |

Relative to BF16, FP16 showed:

- TTFT: approximately 0.31% higher.
- Mean end-to-end latency: approximately 0.23% lower.
- Mean throughput: approximately 0.33% higher.
- Peak allocated GPU memory: effectively unchanged at the reported precision.

These differences are small and are not sufficient to establish a meaningful performance advantage for either precision. Further repetitions and controlled experiments are required.

BF16 remains the reference configuration for subsequent experiments.

## 10. Current Project Status

### Completed

- [x] Created the InferX project structure.
- [x] Configured the Conda environment and PyTorch CUDA support.
- [x] Loaded the local Qwen2.5-1.5B-Instruct model.
- [x] Implemented a baseline inference benchmark.
- [x] Collected initial BF16 and FP16 measurements.
- [x] Implemented repeated latency measurements.
- [x] Recorded TTFT, end-to-end latency, output throughput, and GPU memory.
- [x] Generated CSV result files for analysis.

### Planned

- [ ] Improve token-level timing and TTFT measurement accuracy.
- [ ] Expand workloads across input and output token lengths.
- [ ] Implement controlled batch-size experiments.
- [ ] Evaluate concurrent-request throughput and latency.
- [ ] Investigate quantization and attention optimizations.
- [ ] Compare Transformers with an optimized inference engine, subject to offline dependency availability.
- [ ] Automate experiment summaries and performance plots.
- [ ] Add regression tests and reproducibility checks.

## 11. Reproducibility

For each experiment, record:

- Model identifier and local checkpoint.
- GPU model and allocation details.
- PyTorch, Transformers, and CUDA versions.
- Inference precision and generation parameters.
- Prompt dataset and token counts.
- Warm-up and measured-run counts.
- Raw CSV output and experiment configuration.

Run BF16 and FP16 under equivalent conditions and avoid competing GPU workloads during measurements whenever possible.

Preserve raw CSV files and derive summary tables and plots from those files rather than modifying the original measurements.

## 12. Future Direction

InferX aims to evolve into a reproducible LLM inference benchmarking and optimization framework that compares inference configurations under controlled workloads. The long-term goal is to identify the trade-offs between latency, throughput, GPU memory consumption, and output quality, and to produce evidence-based recommendations for efficient LLM deployment.
