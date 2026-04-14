# Run GEMM benchmark on collected data
# Tests: FP16 baseline, fake quant, real quant (fp32/fp16/int32 accum), custom kernel slot
# Default: test representative layers (q_proj, o_proj, gate_proj, down_proj)

python bench_gemm.py \
    --data_dir ./gemm_data \
    --layers "q_proj,o_proj,gate_proj,down_proj" \
    --batch_sizes "1,2,4" \
    --output gemm_benchmark_results.json
