# Project: ResQ Mixed-Precision Quantization

## Overview

ResQ splits LLM activations into low-variance (4-bit) and high-variance (8-bit) groups after PCA + Hadamard rotation, then uses INT Tensor Core GEMM for inference.

## Directory Structure

```
fake_quant/
  get_basis.py              # Step 0: compute PCA basis
  optimize_rotation.py      # Step 1: optimize rotation matrix
  ptq.py                    # Step 2/3: PTQ evaluation
  collect_gemm_data.py      # Step 4: collect per-layer GEMM data
  bench_gemm.py             # Step 4: GEMM benchmark framework
  collect_activations.py    # activation collection (ResQ original)
  triton_gemm.py            # Triton INT8 GEMM kernels
  csrc/                     # CUTLASS 3.x kernels (WIP)
  utils/                    # quantization utilities
  eval_utils/               # evaluation helpers
  train_utils/              # training helpers
  config_longbench/         # LongBench config files
  0-4_*.sh                  # pipeline shell scripts
```

## Code Hygiene Rules

IMPORTANT: Follow these rules at all times.

### Never commit temporary/debug files

- Do NOT create files named `test_*.py`, `debug_*.py`, `tmp_*.py`, or `scratch_*.py` in the repo
- If you need a throwaway test, write it to `/tmp/` instead
- One-off validation scripts belong in `/tmp/`, not in the project tree

### No duplicate scripts

- Each pipeline step has ONE canonical shell script (0_xxx.sh, 1_xxx.sh, etc.)
- Do NOT create `_v2.sh`, `_real.sh`, `_backup.sh` variants — update the original or use CLI flags
- If a script needs different modes, add a flag (e.g. `--real_quant`) instead of copying the file

### Keep csrc/ clean

- Only one version of each CUDA kernel — no `v1`/`v2` coexistence
- When replacing a kernel, delete the old files in the same commit
- Build scripts: one `setup.py` per kernel module

### Before every commit, check

1. No orphaned test/debug files in the working tree
2. No unused imports in modified Python files
3. No commented-out code blocks longer than 3 lines (delete or use git history)

### bench_gemm.py conventions

- Every test in `ALL_TESTS` must compute its output live (no cloning pre-saved files)
- Every test must have real performance measurement (no borrowing another test's latency)
- When removing a GEMM implementation, also remove its entry from `ALL_TESTS` and dispatch code

## Remote Development

- GPU container: `gemini@general-1295685810-geminijob-0` (H20 Hopper SM90a)
- venv: `source /vllm-workspace/plaquant/.venv/bin/activate`
- CUDA workaround: `LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so`
- GEMM data: `/vllm-workspace/plaquant/project-resq/fake_quant/gemm_data/`
- Ceph backup: `/mnt/gemininjceph3/.../user_spanaluo/plaquant/`

## Key Technical Notes

- Asymmetric quantization: `a_sym=False` means `activation_symmetric=False` (i.e., asymmetric)
- High group (8-bit) uses shift-128 trick: `q_shifted = q_int - 128`, `bias = (128 - zero) * colsum(w)`
- Per-group layers (o_proj): group_k=56, padded to 64 for BLOCK_K alignment
- CUTLASS 3.x on H20 requires `-gencode arch=compute_90a,code=sm_90a` (with `a` suffix)
