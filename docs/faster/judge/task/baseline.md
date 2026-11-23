# Baseline (reference `99764f5`, 2 x B300, 32 CPU cores, per-GPU batch 16)

Frozen timing (`task/timing_ref.json`, 2 runs x (20 warmup + 2 x 100 steps), cold caches):

| | |
|---|---|
| Step time | **906.9 ms** (runs 910.0 / 903.7 ms, spread 0.69%) |
| Benefit threshold | speedup > 1.02 |
| Peak memory | 66.5 GiB per GPU |
| GPU utilization (nvidia-smi, second half of the run) | 53% |
| Time to the first optimizer step | ~165 s (model load, dataset init, DeepSpeed init) |

Shapes per rank and step: 16 samples x 2 images (224 x 224 -> 196 patches -> 49 visual tokens per image after the
2 x 2 merger), about 205 tokens per sample, 4.50 B trainable parameters in 729 tensors, two learning-rate groups.

Main-thread profile of rank 0 (`bench.py profile`, cProfile over 20 steps; the PyTorch profiler and nsys cannot see
kernels on B300, CUPTI has no sm_103 support). Milliseconds per step, cumulative, asynchronous mode / with
`CUDA_LAUNCH_BLOCKING=1` (GPU time charged to the calling frame):

| Stage | async | blocking | Note |
|---|---:|---:|---|
| `build_qwenvl_inputs` (prompt + Qwen3-VL processor) | 190 | 172 | CPU, one thread (torchrun sets OMP_NUM_THREADS=1); GPU idle |
| attention-mask construction (`transformers.masking_utils`, nested vmap) | 154 | 149 | host-bound |
| vision tower (`get_image_features`) | 114 | 119 | ends in a host sync (`.tolist()` of the grid) |
| decoder layers, 36 x | 55 | 111 | |
| DeepSpeed `get_grad_norm_direct` (per-parameter fp64 norms) | 89 | 103 | |
| DeepSpeed gradient all-reduce (10 buckets) | in sync | 245 | overlapped with backward in async mode |
| DeepSpeed optimizer step (fused AdamW on the fp32 partition) | - | 70 | |
| `torch.cuda.synchronize` (DeepSpeed stream/timer syncs: waits for backward + reduction) | 117 | - | |
| data-loader hand-over in the main thread (unpickling PIL batches) | 30 | 30 | the 4 workers keep up: `data_times` ~1 ms |

The backward pass runs on the autograd device thread and is not in the main-thread profile; it is what the
synchronize calls wait for.
