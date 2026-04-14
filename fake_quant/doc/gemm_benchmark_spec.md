# GEMM Benchmark & Data Collection Spec

## 1. 目的

Phase 2 将在 NVIDIA Hopper GPU 上实现自定义 CUTLASS mixed-precision GEMM kernel。在写 kernel 之前需要：

1. **Ground truth 数据**：从真实推理中采集每个 linear layer 的输入/输出/权重，作为 kernel 正确性验证的基准
2. **Benchmark 框架**：测试不同 GEMM 实现（fp16 baseline、fake quant、real quant、未来的 CUDA kernel）的精度和性能
3. **回归测试**：Phase 2-5 每次改动后，用采集的数据验证 kernel 输出是否正确

## 2. 数据采集

### 2.1 数据源

WikiText2 PPL 评估数据集（`eval_utils.evaluator` 跑 PPL 时的输入）。

- 序列长度 2048，真实推理场景
- 数据集固定，结果可复现

### 2.2 Batch Size

采集 batch_size = 1, 2, 4 三种，覆盖不同 M 维度。

### 2.3 采集范围

全部 linear layer 都采集（存储不是问题）。benchmark 测试时选代表层：

| 代表层 | 特点 |
|--------|------|
| `q_proj` | 标准 2048×2048，contiguous [main\|high] split |
| `o_proj` | groupsize=64，contiguous split + column_order permutation |
| `gate_proj` | 2048→8192，大矩阵 |
| `down_proj` | 8192→2048，Hadamard rotation，无 mixed precision |

### 2.4 每层采集的数据

| 数据 | 说明 | 用途 |
|------|------|------|
| FP16 input activation | Hadamard rotation 后、量化前的 x | kernel 输入 |
| 量化后的 activation integers | main group (INT4) + high group (INT8)，含 scale/zero | 验证 activation 量化正确性 |
| FP16 weight | `self.module.weight.data`（GPTQ dequantized） | FP16 baseline 对比 |
| 量化后的 weight integers | checkpoint 直接加载的 Q_int + scale | kernel 输入 |
| Fake quant output | fake quant forward 的输出 | ground truth |
| Real quant output | real quant forward 的输出 | 验证对齐 |
| FP16 baseline output | `x_fp16 @ W_fp16.T`（无量化） | 精度上界 |
| Layer metadata | 层名称、shapes、precision config、column_order | 复现配置 |

### 2.5 采集方式

在 `ActQuantWrapper` 的 forward 中用 `register_forward_hook` 截获中间值：

1. 正常加载模型（fake quant + real quant 双配置）
2. 送入若干 batch WikiText2 数据
3. 每个 `ActQuantWrapper` 上截获输入 x（rotation 后）、fake/real quant 输出
4. 权重数据从已有 buffer（`W_m_int`, `W_m_scale` 等）中读取
5. 保存到磁盘，每层一个目录

### 2.6 输出目录结构

```
gemm_data/
  metadata.json                           # 全局配置：model, w_bits, a_bits, etc.
  layer_00_self_attn_q_proj/
    metadata.json                         # shapes, bits, groupsize, column_order, etc.
    input_fp16_bs1.pt                     # (1, seq, K) rotation后的fp16输入
    input_fp16_bs2.pt                     # (2, seq, K)
    input_fp16_bs4.pt                     # (4, seq, K)
    act_quant_main_bs1.pt                 # {q_int, scale, zero} main group
    act_quant_high_bs1.pt                 # {q_int, scale, zero} high group
    act_quant_main_bs2.pt
    act_quant_high_bs2.pt
    act_quant_main_bs4.pt
    act_quant_high_bs4.pt
    weight_fp16.pt                        # (N, K) GPTQ dequantized fp16
    weight_int_main.pt                    # {q_int, scale} main group (4-bit)
    weight_int_high.pt                    # {q_int, scale} high group (8-bit)
    output_fp16_baseline_bs1.pt           # x_fp16 @ W_fp16.T (无量化)
    output_fp16_baseline_bs2.pt
    output_fp16_baseline_bs4.pt
    output_fake_quant_bs1.pt              # fake quant forward 输出
    output_fake_quant_bs2.pt
    output_fake_quant_bs4.pt
    output_real_quant_bs1.pt              # real quant forward 输出
    output_real_quant_bs2.pt
    output_real_quant_bs4.pt
  layer_00_self_attn_o_proj/
    ...
  layer_00_mlp_gate_proj/
    ...
  ...
```

## 3. Benchmark 测试

### 3.1 测试项目

| # | 测试名称 | 计算方式 | 目的 |
|---|---------|---------|------|
| 1 | FP16 baseline | `x_fp16 @ W_fp16.T` | 无量化理想输出，精度上界 |
| 2 | Fake quant | `fake_quant(x) @ W_dequant.T`（重放采集的 dq_x） | Paper 评估方式，ground truth |
| 3 | Real quant (fp32 accum) | `scale_x * scale_w * (q_x.float() @ q_w.float().T)` | 当前实现，fp32 累加 |
| 4 | Real quant (fp16 accum) | `scale_x * scale_w * (q_x.half() @ q_w.half().T)` | 模拟 fp16 tensor core |
| 5 | Real quant (int32 accum) | `(q_x.int() @ q_w.int().T)` 再乘 scale | 模拟 INT8 tensor core (INT32 累加) |
| 6 | Custom kernel slot | placeholder，加载 `.so` 调用 | Phase 2-5 填入 |

测试 3/4/5 分 main group 和 high group 分别执行后求和：

```
y_m = accumulate(q_x_m, q_w_m) * scale_x_m * scale_w_m   # INT4 × INT4
y_h = accumulate(q_x_h, q_w_h) * scale_x_h * scale_w_h   # INT8 × INT8
y = y_m + y_h
```

### 3.2 精度 Metrics

所有 metric 与两个 reference 分别比较：FP16 baseline 和 fake quant output。

| Metric | 公式 | 意义 |
|--------|------|------|
| Max Absolute Error | `max(\|y_test - y_ref\|)` | 最坏情况误差 |
| Mean Absolute Error (MAE) | `mean(\|y_test - y_ref\|)` | 平均绝对误差 |
| RMSE | `sqrt(mean((y_test - y_ref)²))` | L2 归一化误差 |
| MAPE | `mean(\|y_test - y_ref\| / (\|y_ref\| + ε))` | 相对误差 |
| Cosine Similarity | `cos(y_test, y_ref)` | 方向一致性 |
| SNR (dB) | `10 * log10(‖y_ref‖² / ‖y_test - y_ref‖²)` | 信噪比 |

### 3.3 性能 Metrics

| Metric | 说明 |
|--------|------|
| Latency (ms) | 单次 GEMM 耗时，warmup 10 次后取 100 次平均 |
| TFLOPS | `2 * M * N * K / latency / 1e12` |
| Throughput vs FP16 | `latency_fp16 / latency_test`，加速比 |
| Memory (MB) | 权重存储大小（int4 vs int8 vs fp16） |

### 3.4 输出格式

终端表格 + JSON 文件：

```
Layer: model.layers.0.self_attn.q_proj (M=2048, N=2048, K=1792+256)
BatchSize=1:
┌──────────────────┬──────────┬────────┬────────┬──────────┬─────────┬────────┬────────┬────────┐
│ Test             │ MaxAbsErr│ MAE    │ RMSE   │ CosineSim│ SNR(dB) │Lat(ms) │ TFLOPS │ vs FP16│
├──────────────────┼──────────┼────────┼────────┼──────────┼─────────┼────────┼────────┼────────┤
│ FP16 baseline    │ 0.0000   │ 0.0000 │ 0.0000 │ 1.000000 │ inf     │ 0.42   │ 40.5   │ 1.00x  │
│ Fake quant       │ 0.1523   │ 0.0089 │ 0.0124 │ 0.999847 │ 38.2    │ 0.43   │ 39.8   │ 0.98x  │
│ Real (fp32 acc)  │ 0.1524   │ 0.0089 │ 0.0124 │ 0.999847 │ 38.2    │ 0.45   │ 38.1   │ 0.93x  │
│ Real (fp16 acc)  │ 0.1530   │ 0.0091 │ 0.0127 │ 0.999842 │ 38.0    │ 0.42   │ 40.5   │ 1.00x  │
│ Real (int32 acc) │ 0.1524   │ 0.0089 │ 0.0124 │ 0.999847 │ 38.2    │ 0.20   │ 85.3   │ 2.10x  │
│ Custom kernel    │   -      │   -    │   -    │   -      │   -     │   -    │   -    │   -    │
└──────────────────┴──────────┴────────┴────────┴──────────┴─────────┴────────┴────────┴────────┘
```

## 4. 脚本

| 脚本 | 功能 |
|------|------|
| `collect_gemm_data.py` | 数据采集：加载模型 → 跑推理 → hook 截获 → 保存到磁盘 |
| `bench_gemm.py` | Benchmark：加载磁盘数据 → 跑各种 GEMM → 计算 metrics → 输出表格 |
| `4_collect_gemm.sh` | 采集脚本的 shell wrapper（torchrun） |
| `4_bench_gemm.sh` | Benchmark 脚本的 shell wrapper（单 GPU，不需要 torchrun） |
