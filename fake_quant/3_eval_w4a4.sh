# W4A4KV4 evaluation with GPTQ weight quantization
# This is the full ResQ paper configuration

torchrun --nnodes=1 --nproc_per_node=1 --master_port=24544 ptq.py \
--input_model unsloth/Llama-3.2-1B-Instruct \
--per_device_eval_batch_size 1 \
--model_max_length 2048 \
--fp16 True \
--bf16 False \
--w_bits 4 \
--a_bits 4 \
--k_bits 4 \
--v_bits 4 \
--high_bits 8 \
--low_bits 2 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 64 \
--v_groupsize 64 \
--high_fraction 0.125 \
--low_fraction 0.0 \
--rotate_mode "resq" \
--optimized_rotation_path ./rotation/R-high-0.125-low-0.0-sparse-0.0-Llama-3.2-1B.bin \
--optimized_basis_path ./rotation/U-wikitext-512-Llama-3.2-1B.bin \
--rotation_granularity 'full_shared' \
--rotate \
--tasks "boolq" \
--save_qmodel_path ./qmodels/W4A4KV4-Llama-3.2-1B.pt
