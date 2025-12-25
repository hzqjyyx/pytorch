这个文件定义了 Flash Attention 在 ROCm/HIP 平台上的 API 接口。

## 核心功能

### 主函数签名
文件提供了 Multi-Head Attention (MHA) 的前向计算接口，主要有两个变体：

1. **标准版本** `mha_fwd` (260-323行)
   - 输入：Q, K, V 张量，形状为 `batch_size x seqlen x num_heads x head_size`
   - 输出：8个张量的元组（注意力输出及中间结果）
   - 支持参数：dropout、softmax_scale、causal masking、sliding window、softcap

2. **变长序列版本** `mha_varlen_fwd` (334-429行)  
   - 输入：Q, K, V 张量，形状为 `total_q/k x num_heads x head_size`（packed sequences）
   - 额外输入：`cu_seqlens_q/k`（累积序列长度，用于区分 batch 中各序列边界）
   - 适用于不同序列长度的批处理场景

### 关键特性
- **Attention bias**: 通过 `alibi_slopes_` 支持 ALiBi 位置编码
- **Causal masking**: `is_causal` 参数支持自回归场景
- **Sliding window**: `window_size_left/right` 实现局部注意力
- **Dropout**: `p_dropout` + 随机数生成器 `gen_`
- **可选输出**: `return_softmax` 控制是否返回 softmax 结果

### 实现细节
函数通过条件编译选择后端实现：
- 检查 `USE_CK_FLASH_ATTENTION` 宏和运行时配置
- 分发到 `mha_fwd_aot` (AOTriton) 或 `mha_fwd_ck` (Composable Kernel)
- 两个后端接口略有差异（如 CK 支持 `attn_bias_`，AOTriton 支持 `alibi_slopes_`）

---

**忽略的内容简述：**
- ROCm 后端选择逻辑（CK vs AOTriton）
- Backward 函数：`mha_bwd`, `mha_varlen_bwd` 及其变体
- CK 专有函数签名（`mha_*_ck` 系列，支持 bias gradients）
