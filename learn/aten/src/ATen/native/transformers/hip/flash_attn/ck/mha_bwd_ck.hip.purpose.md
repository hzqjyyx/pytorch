这个文件实现了 Flash Attention 的反向传播（backward pass）功能，用于计算梯度。主要流程：

## 核心函数

**`mha_bwd_ck`** (215-443行) - 主入口函数
- 接收前向传播的输出和梯度，计算输入张量的梯度
- 输入：`dout`(输出梯度), `q/k/v`(查询/键/值), `out`(前向输出), `softmax_lse`
- 输出：`dq, dk, dv`(三个输入张量的梯度), `softmax_d`, `dbias`

## 主要步骤

1. **参数验证** (240-286行)
   - 检查数据类型（仅支持 fp16/bf16）
   - 验证张量形状和连续性
   - 检查 head_size 是8的倍数且 ≤128

2. **掩码处理** (288-300行)
   - 根据 `is_causal`、`window_size_left/right` 构造掩码信息
   - 支持三种模式：causal、no mask、local window

3. **内存分配** (310-368行)
   - 创建梯度张量 `dq, dk, dv`（如果未提供）
   - 分配 `softmax_d` 和 `dq_accum`（累加器）
   - 对 MQA/GQA 创建扩展的 `dk_expanded, dv_expanded`

4. **调用 CK kernel** (376-421行)
   - 构造 `traits` 和 `args` 配置
   - 调用底层 `fmha_bwd` 执行实际计算
   - 特殊处理 `seqlen_q == 0` 的边界情况

5. **后处理** (423-439行)
   - 对 MQA/GQA，将多头梯度求和回原始头数
   - 裁剪填充的维度（如果 head_size 不是8的倍数）

## 辅助函数

**`get_ck_fmha_bwd_traits`** (11-29行)
- 构造反向传播的特征配置
- 包含：head_size、数据类型、掩码类型、bias设置、dropout、确定性模式

**`get_ck_fmha_bwd_args`** (31-214行)
- 提取所有张量的步长（stride）信息
- 处理 attention bias 和 gradient bias 的指针和步长
- 构造完整的参数结构传递给底层 kernel

---

**ROCm/HIP 相关**：
- 使用 HIP stream 和 `at::cuda::getCurrentHIPStream()`
- `at::hip::HIPGuardMasqueradingAsCUDA` 设备保护

**Backward 细节**：
- 支持确定性模式（deterministic）通过调整 `dq_accum` 的 split 数量
- 支持 dropout（通过 philox seed/offset）
- 支持 bias 梯度计算（`dbias`）
- 对 `dq` 使用原子操作，需要先清零
