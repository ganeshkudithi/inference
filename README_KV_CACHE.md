# KV Cache Optimization Results

## 1. Overview

This experiment evaluates the impact of enabling and disabling the
**Key-Value (KV) cache** during autoregressive inference with the
Qwen2.5-1.5B-Instruct model.

The objectives are to measure throughput, compare static batch sizes,
and verify whether KV caching changes greedy-generation outputs.

## 2. Experimental Environment

  -----------------------------------------------------------------------
  Setting                             Value
  ----------------------------------- -----------------------------------
  Model                               Qwen2.5-1.5B-Instruct

  GPU                                 NVIDIA A100 80GB PCIe

  Precision                           FP32

  Attention implementation            SDPA

  TF32                                Disabled

  PyTorch                             2.6.0+cu124

  Transformers                        4.57.1

  Maximum new tokens                  128

  Number of requests                  24

  Batch sizes                         1, 2, 4, 8

  Benchmark repetitions               5 runs per batch size

  Generation strategy                 Greedy decoding (`do_sample=False`,
                                      `num_beams=1`)
  -----------------------------------------------------------------------

The workload contains three prompts (short, medium, and long), repeated
in the same order to form 24 requests. The same model, workload,
precision, attention implementation, and TF32 setting were used for both
cache configurations.

## 3. KV Cache: ON vs OFF

During autoregressive generation, a language model generates one token
at a time.

-   **KV cache ON:** Previously computed attention keys and values are
    retained and reused during subsequent decoding steps.
-   **KV cache OFF:** Previously computed keys and values are not reused
    through a persistent generation cache, so attention-related
    computation is repeated during decoding.

KV caching trades additional memory for reduced repeated computation. It
does not retrain or modify the model weights.

## 4. Throughput Results

The following values are the mean throughput across five benchmark runs
for each batch size.

    Batch size   KV cache ON (tok/s)   KV cache OFF (tok/s)   Speedup
  ------------ --------------------- ---------------------- ---------
             1                45.794                 20.958     2.19×
             2                65.474                 20.345     3.22×
             4               128.596                 22.077     5.83×
             8               252.621                 23.889    10.57×

Speedup is calculated as:

\[ `\text{Speedup}`{=tex} =
`\frac{\text{Throughput with KV cache ON}}`{=tex}
{`\text{Throughput with KV cache OFF}`{=tex}} \]

### Observations

1.  KV caching improved measured throughput at every tested batch size.
2.  The measured benefit increased with batch size in this workload.
3.  At batch size 8, throughput increased from 23.889 tok/s to 252.621
    tok/s, corresponding to approximately 10.57× speedup.
4.  The throughput comparison used the same total generated-token count
    per run (2,448 tokens), making the two cache configurations directly
    comparable for this workload.

## 5. Memory Results

Mean peak allocated GPU memory from the benchmark runs:

    Batch size   KV cache ON (GiB)   KV cache OFF (GiB)
  ------------ ------------------- --------------------
             1               5.782                5.793
             2               5.796                5.821
             4               5.826                5.876
             8               5.886                5.985

Peak allocated memory was slightly higher with KV caching disabled in
these measurements. The difference was small compared with total
allocated memory.

**Note:** PyTorch allocated memory and reserved memory are different
measurements. Reserved memory includes memory held by PyTorch's caching
allocator and should not be interpreted as memory actively occupied by
model tensors.

## 6. Correctness Validation

Correctness was evaluated using deterministic greedy generation.
Generated token IDs were compared exactly.

### 6.1 Batch-size consistency with KV cache ON

Each batch-size result was compared against the batch-size-1 reference
generated with KV caching enabled.

    Batch size   Exact matches   Match rate
  ------------ --------------- ------------
             1           24/24         100%
             2           24/24         100%
             4           24/24         100%
             8           24/24         100%

### 6.2 Batch-size consistency with KV cache OFF

Each batch-size result was compared against the batch-size-1 reference
generated with KV caching disabled.

    Batch size   Exact matches   Match rate
  ------------ --------------- ------------
             1           24/24         100%
             2           24/24         100%
             4           24/24         100%
             8           24/24         100%

### 6.3 Direct comparison: KV cache ON vs OFF

The batch-size-1 outputs from the two cache configurations were compared
request by request.

  Comparison                         Exact matches   Match rate
  -------------------------------- --------------- ------------
  KV ON vs KV OFF (batch size 1)             24/24         100%

All 24 requests produced identical token-ID sequences with KV caching
enabled and disabled.

### Correctness interpretation

For this model, numerical configuration, and 24-request workload:

-   Changing batch size did not change the generated token IDs in either
    cache configuration.
-   Enabling versus disabling KV caching did not change the generated
    token IDs in the direct batch-size-1 comparison.

These results establish exact agreement for the tested cases. They do
not prove equivalence for every possible prompt, sequence length, model,
precision, or hardware configuration.

## 7. Commands to Reproduce

Run these commands from the InferX project root, inside the configured
environment and a Slurm GPU allocation.

### Benchmark with KV cache ON

``` bash
python scripts/benchmark_batch.py \
    --config configs/fp32.json \
    --attention sdpa \
    --tf32 off \
    --kv-cache on
```

### Benchmark with KV cache OFF

``` bash
python scripts/benchmark_batch.py \
    --config configs/fp32.json \
    --attention sdpa \
    --tf32 off \
    --kv-cache off
```

### Correctness validation with KV cache ON

``` bash
python scripts/validate_batch_correctness.py \
    --config configs/fp32.json \
    --attention sdpa \
    --tf32 off \
    --kv-cache on \
    --batch-sizes 1 2 4 8 \
    --requests 24
```

### Correctness validation with KV cache OFF

``` bash
python scripts/validate_batch_correctness.py \
    --config configs/fp32.json \
    --attention sdpa \
    --tf32 off \
    --kv-cache off \
    --batch-sizes 1 2 4 8 \
    --requests 24
```

### Direct comparison of KV cache outputs

``` bash
python scripts/compare_kv_cache_correctness.py
```

The direct comparison script uses the two detailed correctness CSV files
produced by the corresponding validation runs. Update the file paths in
that script if the CSV filenames differ.

## 8. Result Files

The experiment outputs were saved in `results/`.

  --------------------------------------------------------------------------------------------------
  File                                                           Description
  -------------------------------------------------------------- -----------------------------------
  `results/processed/correctness_summary_20261106T114130Z.csv`   KV ON correctness summary

  `results/processed/correctness_details_20261106T114130Z.csv`   KV ON per-request token results

  `results/processed/correctness_summary_20261106T115412Z.csv`   KV OFF correctness summary

  `results/processed/correctness_details_20261106T115412Z.csv`   KV OFF per-request token results

  `results/processed/correctness_details_20261106T120143Z.csv`   KV ON detailed results used in
                                                                 direct comparison

  `results/processed/correctness_details_20261106T121031Z.csv`   KV OFF detailed results used in
                                                                 direct comparison
  --------------------------------------------------------------------------------------------------

Keep the original benchmark CSV files in `results/raw/` as the source
data for the throughput and memory measurements.

## 9. Conclusion

KV caching substantially improved inference throughput for
Qwen2.5-1.5B-Instruct on the NVIDIA A100 80GB GPU under the tested FP32
and SDPA configuration. The measured speedup increased from 2.19× at
batch size 1 to 10.57× at batch size 8.

All tested correctness comparisons achieved 100% exact token matching:
across batch sizes with KV caching enabled, across batch sizes with KV
caching disabled, and directly between the enabled and disabled cache
configurations at batch size 1.

**Conclusion:** KV caching is a strong inference optimization for this
tested workload, delivering substantially higher throughput without
observed changes to generated token IDs.

Further validation on additional prompts, longer sequences, different
precisions, and other model sizes is needed before generalizing these
results.
