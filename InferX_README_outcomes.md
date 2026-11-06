# InferX: LLM Inference Optimization and Benchmarking Framework

**Status:** Experimental benchmarking prototype\
**Hardware:** NVIDIA A100 80GB PCIe\
**Model:** Qwen2.5-1.5B-Instruct\
**Purpose:** Evaluate the trade-offs between LLM inference throughput,
latency, resource utilization, and output correctness.

## 1. Project overview

InferX is an experimental framework for benchmarking and analyzing large
language model inference configurations on GPU systems. It studies how
runtime settings affect generation throughput, latency, GPU memory, GPU
utilization, power consumption, and generated-token consistency.

The current experiments use Qwen2.5-1.5B-Instruct on an NVIDIA A100 80GB
GPU. This is an inference optimization project: model weights are not
being trained or updated. The experiments change how the existing model
executes.

The longer-term goal is to compare configurations systematically and
eventually recommend suitable inference settings for a given workload
and correctness requirement.

## 2. Experimental environment

  Component                 Current setup
  ------------------------- -----------------------
  Operating system          AlmaLinux 8.9
  GPU                       NVIDIA A100 80GB PCIe
  NVIDIA driver             550.54.15
  CUDA reported by driver   12.4
  Python                    3.11.17
  PyTorch                   2.6.0+cu124
  Transformers              4.57.1
  Model                     Qwen2.5-1.5B-Instruct
  Execution                 Slurm GPU allocation
  Model loading             Local/offline

Model directory: `models/Qwen2.5-1.5B-Instruct`

Offline environment:

``` bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 3. Experimental dimensions

  -----------------------------------------------------------------------
  Dimension               Values investigated     Purpose
  ----------------------- ----------------------- -----------------------
  Precision               FP32, FP16, BF16        Performance and
                                                  numerical consistency

  Attention               SDPA, Eager             Compare attention
                                                  backends

  TF32                    ON, OFF                 Evaluate TF32 during
                                                  FP32 execution

  Static batch size       1, 2, 4, 8              Measure batching
                                                  benefits

  KV cache                ON/OFF in selected      Investigate generation
                          tests                   behavior

  Prompt workload         Short, medium, long     Study workload
                                                  variation

  Correctness             Exact generated-token   Compare against
                          match                   batch-size-1 reference
  -----------------------------------------------------------------------

The workload is stored in `workloads/prompts.jsonl`. It contains short,
medium, and long prompts. The benchmark workload cycles through these
prompt types to form 24 requests.

## 4. Current framework capabilities

### 4.1 Baseline benchmarking

Single-request inference baselines have been measured for FP32, FP16,
and BF16, including generation performance and GPU memory observations.

### 4.2 Static batching

The batch benchmark evaluates batch sizes 1, 2, 4, and 8. It supports
precision configuration files, SDPA/Eager attention, TF32 ON/OFF,
repeated runs, warm-up runs, throughput, and memory measurements.

### 4.3 Correctness validation

`validate_batch_correctness.py` compares generated token sequences from
batched execution against the batch-size-1 reference. It uses greedy
decoding (`do_sample=False`) and supports SDPA and Eager attention.

Exact token matching is a strict reproducibility metric: one changed
token makes a sequence a mismatch. A mismatch does not by itself prove
that an output is semantically incorrect.

### 4.4 Numerical diagnostics

Diagnostic scripts have been used to inspect logits, top-token rankings,
decoding-step divergence, TF32 effects, and FP16 Eager numerical
behavior.

### 4.5 Resource monitoring

GPU monitoring has captured utilization, memory usage, power draw, and
temperature using `nvidia-smi`. CPU process/thread monitoring has been
started using `pgrep` and `top`.

## 5. Experimental results and outcomes

Results below apply to the current model, GPU, workload, and benchmark
implementation. They are not universal performance guarantees.

### 5.1 Baseline precision comparison

Representative single-request throughput observations:

| Precision | Short | Medium | Long |
|-----------|------:|-------:|-----:|
| BF16      | ~53.44 tok/s | ~53.73 tok/s | ~53.62 tok/s |
| FP16      | ~52.88 tok/s | ~53.07 tok/s | ~52.99 tok/s |

An earlier BF16 run measured approximately 52.3--52.8 tok/s, indicating
some run-to-run variation.

**Outcome:** FP16 did not show a meaningful speed advantage over BF16 in
these measurements. The observed differences are small.

### 5.2 Static batching throughput

Repeated FP32 benchmark summaries using SDPA and TF32 OFF:

| Batch Size | Mean Aggregate Throughput |
|-----------:|--------------------------:|
| 1          | ~49.45 tokens/s |
| 2          | ~70.71 tokens/s |
| 4          | ~140.52 tokens/s |
| 8          | ~269.50 tokens/s |

Batch size 8 achieved approximately **5.45×** the aggregate throughput
of batch size 1.

### Batch Size vs Mean Aggregate Throughput

![Batch Size vs Throughput](figures/batch_throughput.png)

**Figure 1:** Mean aggregate inference throughput for different batch sizes.

A separate BF16 experiment measured:

| Batch Size | Throughput |
|-----------:|-----------:|
| 1 | 53.68 tokens/s |
| 2 | 66.62 tokens/s |
| 4 | 131.51 tokens/s |
| 8 | 269.43 tokens/s |

**Outcome:** Static batching substantially improves aggregate throughput
in this workload. Some BF16 runs generated different total token counts
across batch sizes, so throughput must be interpreted together with
generated-token counts and workload equivalence.

**Important:** Aggregate throughput is not individual request latency.
Static batching may increase the time a request waits for a batch. These
experiments do not yet establish online serving latency under dynamic
arrivals.

### 5.3 FP32 TF32 ON versus OFF

Observed throughput:

| Batch Size | TF32 OFF (tok/s) | TF32 ON (tok/s) | Change with TF32 ON |
|-----------:|----------------:|---------------:|--------------------:|
| 1 | 49.65 | 48.40 | -2.51% |
| 2 | 71.22 | 68.31 | -4.07% |
| 4 | 139.75 | 138.58 | -0.83% |
| 8 | 271.33 | 276.91 | +2.06% |

Exact-token matches against the batch-size-1 reference:

| Batch Size | TF32 OFF | TF32 ON |
|-----------:|---------:|--------:|
| 1 | 24/24 | 24/24 |
| 2 | 24/24 | 24/24 |
| 4 | 24/24 | 24/24 |
| 8 | 24/24 | 16/24 |

**Outcome:** TF32 did not provide a consistent throughput improvement.
TF32 ON at batch size 8 reduced exact-match consistency to 16/24. FP32
with TF32 OFF is the safer reference setting for the current workload.

### 5.4 SDPA versus Eager attention

Repeated FP32, TF32 OFF results:

| Batch Size | SDPA (tok/s) | Eager (tok/s) | SDPA Improvement |
|-----------:|-------------:|-------------:|-----------------:|
| 1 | ~49.45 | ~45.21 | ~9.4% |
| 2 | ~70.71 | ~70.05 | ~0.9% |
| 4 | ~140.52 | ~139.00 | ~1.1% |
| 8 | ~269.50 | ~269.37 | ~0.05% |

FP32 correctness validation produced 24/24 exact matches for both SDPA
and Eager at batch sizes 1, 2, 4, and 8.

**Outcome:** SDPA was clearly faster at batch size 1 in these runs. At
larger batch sizes, the throughput difference was small. Both backends
passed exact-token validation for the tested FP32 workload.

### 5.5 Correctness across precision and batch size

Selected exact-token results against the batch-size-1 reference:

  Configuration         Batch size   Exact matches
  ------------------- ------------ ---------------
  BF16, SDPA, KV ON              1           24/24
  BF16, SDPA, KV ON              2            0/24
  BF16, SDPA, KV ON              4            8/24
  BF16, SDPA, KV ON              8            8/24
  FP16                           1           24/24
  FP16                           2           12/24
  FP16                           4           16/24
  FP16                           8            8/24

Other BF16 diagnostics found mismatches with KV cache disabled and with
Eager attention under different experiment settings; those results
should not be treated as one controlled comparison.

FP32 with TF32 OFF and SDPA achieved 24/24 exact matches at all tested
batch sizes. FP32 Eager also achieved 24/24 at all tested batch sizes.

**Outcome:** Precision and execution configuration can affect exact
generated-token consistency. Correctness evaluation is therefore
necessary alongside performance benchmarking.

### 5.6 Logit-level numerical investigation

Selected diagnostics found:

-   BF16 SDPA comparisons between individual and batched prompt
    execution had maximum absolute logit differences of approximately
    0.25--0.31 in the examined prefill cases, while top-1 predictions
    remained the same in those particular comparisons.
-   In a BF16 decoding diagnostic with identical prompt and generated
    history, the first three next-token predictions matched, but a later
    step selected different top tokens.
-   In one FP32 decoding-step comparison, TF32 OFF produced a maximum
    absolute logit difference of approximately `7.63e-05`; TF32 ON
    produced approximately `0.02697`.

**Outcome:** Small numerical changes can alter token rankings at later
decoding steps. The precise GPU kernel or operation responsible has not
been conclusively isolated.

### 5.7 FP16 Eager numerical issue

A diagnostic experiment found:

-   FP16 Eager attention produced NaNs in final logits.
-   FP16 SDPA produced finite final logits in the tested case.
-   BF16 Eager produced finite values in the tested case.
-   NaNs were first observed around the attention output projection
    (`o_proj`) in the investigated FP16 Eager execution.

**Outcome:** FP16 Eager requires further investigation and should not be
assumed safe from this limited test. The exact numerical mechanism has
not been conclusively identified.

### 5.8 GPU and CPU resource observations

A GPU monitoring run on the A100 showed:

  Metric                                                            Observation
  ---------------------------------------------------- ------------------------
  GPU used for inference                                                  GPU 0
  Other GPU                                                 GPU 1 remained idle
  GPU memory during sustained execution                               \~6.5 GiB
  GPU utilization during much of sustained execution                  \~44--53%
  GPU power during much of sustained execution                     \~109--120 W
  GPU temperature                                        Approximately 45--60°C

A separate Python process snapshot showed approximately 99.7% CPU usage,
corresponding to roughly one logical CPU core at that instant. A later
`top -H` command showed zero threads because the benchmark process had
already exited.

**Outcome:** The tested workload uses a small fraction of the A100
memory capacity. GPU utilization was moderate during much of the
observed run. These observations motivate further profiling but do not
prove CPU bottlenecking. Simultaneous CPU and GPU monitoring during the
same active benchmark remains necessary.

## 6. Current reference configuration

For controlled comparisons, the current reference configuration is:

  Setting                             Reference
  ----------------------------------- ----------------------------
  Precision                           FP32
  TF32                                OFF
  Attention                           SDPA
  KV cache                            ON
  Correctness reference batch size    1
  Throughput experiment batch sizes   1, 2, 4, 8
  Decoding                            Greedy (`do_sample=False`)

This is a reference for the current experiments, not a claim that FP32
is always the best production configuration.

## 7. Repository layout

Representative project structure based on components used so far:

``` text
InferX/
├── configs/
│   ├── fp32.json
│   ├── fp16.json
│   └── bf16.json
├── docs/
├── models/
│   └── Qwen2.5-1.5B-Instruct/
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
├── scripts/
│   ├── benchmark_batch.py
│   ├── validate_batch_correctness.py
│   ├── benchmark_latency.py
│   ├── debug_batch_logits.py
│   ├── debug_decode_steps.py
│   ├── debug_step4_logits.py
│   ├── compare_batch_prefill.py
│   └── debug_batch_positions.py
├── src/
├── tests/
├── workloads/
│   └── prompts.jsonl
└── README.md
```

This is representative, not a guaranteed exhaustive listing of the
current checkout. Confirm filenames before relying on every item being
present. Model files and generated result directories may be excluded
from Git.

## 8. Running experiments

### Activate environment and enable offline loading

``` bash
conda activate inferx
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Run GPU workloads inside an active Slurm GPU allocation.

### Run batch benchmark

``` bash
python scripts/benchmark_batch.py \
    --config configs/fp32.json \
    --batch-sizes 1 2 4 8 \
    --tf32 off \
    --attention sdpa
```

For Eager attention:

``` bash
python scripts/benchmark_batch.py \
    --config configs/fp32.json \
    --batch-sizes 1 2 4 8 \
    --tf32 off \
    --attention eager
```

For TF32 ON:

``` bash
python scripts/benchmark_batch.py \
    --config configs/fp32.json \
    --batch-sizes 1 2 4 8 \
    --tf32 on \
    --attention sdpa
```

### Run correctness validation

``` bash
python scripts/validate_batch_correctness.py \
    --config configs/fp32.json \
    --batch-sizes 1 2 4 8 \
    --attention sdpa
```

### Monitor GPU

``` bash
nvidia-smi
```

One-second time-series monitoring:

``` bash
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu \
  --format=csv \
  -l 1
```

Redirect output to a uniquely named file under `results/raw/` to save
it.

### Monitor benchmark process

``` bash
pgrep -af benchmark_batch.py
top -H -p <PID>
```

Start monitoring while the benchmark is active. A process ID will no
longer be available after the benchmark exits.

## 9. Reproducibility and interpretation guidelines

1.  Compare like with like: control precision, attention backend, TF32,
    KV cache, batch size, workload, and generated-token count.
2.  Separate aggregate throughput from per-request latency.
3.  Use repeated runs and report variation, not only the best run.
4.  Include warm-up runs.
5.  Record generated-token counts alongside throughput.
6.  Treat exact token mismatch as a strict reproducibility difference,
    not a direct semantic-quality score.
7.  Do not infer a bottleneck from one GPU-utilization snapshot.
8.  Capture CPU and GPU metrics simultaneously during active execution.
9.  Record software versions, hardware, configuration, workload, run
    count, warm-up count, and result filenames.

## 10. Current limitations

-   Experiments primarily use one 1.5B model and one A100 GPU.
-   Workloads are controlled prompts rather than representative
    production request traces.
-   Static batching has been tested; continuous batching is not yet
    implemented or evaluated.
-   TTFT and ITL measurement need further validation. The current
    latency benchmark estimates ITL rather than directly measuring every
    token interval.
-   Quantization experiments have not yet been completed.
-   CPU/GPU bottleneck attribution is not conclusive.
-   Exact-token comparison does not replace semantic-quality evaluation.
-   The precise kernel-level cause of observed numerical differences has
    not been isolated.

## 11. Next development milestones

### Priority 1 --- Complete resource profiling

-   Capture CPU process/thread utilization and GPU utilization
    simultaneously.
-   Repeat monitoring for batch sizes 1, 2, 4, and 8.
-   Determine whether GPU utilization changes with batch size.
-   Investigate CPU-side preprocessing, synchronization, and
    kernel-launch overhead only when supported by measurements.

### Priority 2 --- Improve latency measurement

-   Measure TTFT separately from generation time.
-   Improve per-token ITL measurement.
-   Report latency distributions such as median and p95.
-   Distinguish batch execution latency from per-request latency.

### Priority 3 --- Expand inference optimization

-   Evaluate supported INT8 and 4-bit inference paths.
-   Investigate KV-cache memory and performance behavior.
-   Implement and benchmark continuous batching.
-   Test additional prompt lengths and request arrival patterns.

### Priority 4 --- Automated configuration analysis

-   Store configuration and result metadata consistently.
-   Evaluate performance and correctness together.
-   Define workload-specific acceptance constraints.
-   Rank or recommend configurations automatically.

### Priority 5 --- Generalization

-   Test additional model sizes and architectures where resources
    permit.
-   Evaluate more workloads and request distributions.
-   Repeat experiments to establish measurement variability.

## 12. Current conclusion

The experiments completed so far establish that:

1.  Static batching can substantially improve aggregate throughput for
    the tested workload.
2.  The performance benefit of attention backends depends on batch size.
3.  TF32 did not provide a consistent throughput improvement in the
    tested configuration.
4.  Precision and batch shape can affect exact generated-token
    consistency.
5.  Correctness validation is necessary alongside performance
    benchmarking.
6.  The workload uses substantially less than the A100's available
    memory, while moderate GPU utilization motivates further profiling.

InferX has established a practical foundation for systematic LLM
inference benchmarking. The next step is to progress from individual
configuration comparisons toward resource-aware profiling, broader
optimization experiments, and automated configuration selection.

## 13. Recommended metadata for every reported experiment

Record the following with each result:

-   Model and model revision
-   GPU model
-   Precision
-   Attention implementation
-   TF32 setting
-   KV-cache setting
-   Batch size
-   Workload and request count
-   Warm-up and measured run counts
-   Generated-token count
-   Throughput and latency definitions
-   Correctness reference and exact-match count
-   Result file path
