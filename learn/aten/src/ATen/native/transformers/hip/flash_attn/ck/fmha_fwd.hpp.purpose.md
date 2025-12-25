这是 AMD ROCm 平台上 Flash Attention 前向传播的核心接口文件，基于 Composable Kernel (CK) 库实现。主要功能：

## 数据类型配置

定义了多种精度的 FMHA 类型配置（FmhaFwdTypeConfig）：
- **FP16/BF16**: 标准半精度类型，各组件（Q/K/V/Bias/O）使用相同精度
- **FP8/BF8**: 8位浮点，Bias 使用 float 以保持精度
- 所有配置统一使用 float 作为：
  - LSE (log-sum-exp) 累加
  - 第一次 GEMM (Q×K^T) 的累加器
  - Softmax 计算
  - 第二次 GEMM (P×V) 的累加器

## 注意力掩码类型

通过 FmhaMasks 提供三种掩码模式：
- **NoMask**: 无掩码
- **GenericMask**: 通用掩码（支持任意模式）
- **CausalMask**: 因果掩码（自回归场景）

## 核心运行时参数结构

**fmha_fwd_args** - 标准前向传播：
- 输入输出指针：Q/K/V、bias、dropout random values、LSE、输出 O
- 变长序列支持：seqstart_q/k_ptr、seqlen_k_ptr（用于 group mode）
- 形状参数：seqlen_q/k、batch、hdim_q/v、nhead_q/k（支持 GQA/MQA）
- 缩放因子：scale_s (QK^T)、scale_p (softmax后)、scale_o (输出)
- 跨步参数：支持非连续内存布局的完整 stride 配置
- 滑动窗口注意力：window_size_left/right
- Dropout：p_drop、seed/offset

**fmha_fwd_splitkv_args** - Split-KV 优化版本：
- 用于处理超长序列，将 K/V 沿序列维度切分
- 额外参数：
  - num_splits：切分数量
  - lse_acc_ptr/o_acc_ptr：中间累加缓冲区
  - block_table_ptr：分页 KV cache 支持
  - cache_batch_idx：批次索引映射
  - is_gappy：区分连续/非连续 seqstart_k 使用模式

**fmha_fwd_appendkv_args** - KV Cache 追加：
- 用于增量解码场景
- knew_ptr/vnew_ptr：新的 K/V 值
- rotary_cos/sin_ptr：RoPE 旋转位置编码
- block_table 支持分页 KV cache

## 内核配置特征 (Traits)

**fmha_fwd_traits_** - 编译时模板参数：
- 块切分参数：kM0/kN0/kK0 (第一次GEMM)、kN1/kK1 (第二次GEMM)
- kK0BlockLength：K 维度的块长度
- FmhaPipelineEnum：流水线策略
- 填充标志：kPadS/kPadSK/kPadD/kPadDv（处理非对齐维度）
- kDoFp8StaticQuant：FP8 静态量化开关

**fmha_fwd_splitkv_traits_** - Split-KV 特征：
- 增加 kIsPagedKV：分页 KV cache 支持

**fmha_fwd_appendkv_traits_** - AppendKV 特征：
- kTileSizeS/Sk/D/Dv：各维度的 tile 大小
- RotaryEnum：RoPE 类型枚举

## 核心函数接口

**fmha_fwd_create_kargs_and_grids**：
- 根据 batch/group mode 创建内核参数（kargs）
- 计算 GPU grid 维度（基于 batch、nhead、seqlen、hdim）
- Group mode 支持变长序列（通过 seqstart 指针）

**fmha_fwd_splitkv_create_kargs_and_grids**：
- Split-KV 主内核的参数/网格创建
- 处理分页 KV cache 和 gappy 模式

**fmha_fwd_splitkv_combine_create_kargs_and_grids**：
- Split-KV 合并阶段：将多个 split 的 LSE 和 O 合并

**fmha_fwd_appendkv_create_kargs_and_grids**：
- AppendKV 内核参数/网格创建

**实际执行函数**：
- `fmha_fwd_()`: 标准前向传播实现
- `fmha_fwd_splitkv_oneshot_()`: Split-KV 一次性执行
- `fmha_fwd_splitkv_combine_oneshot_()`: Split-KV 合并
- `fmha_fwd_appendkv_()`: KV cache 追加

## 公共 API

**fmha_fwd** - 运行时分发：
- 输入 traits（hdim、数据类型、掩码类型等）
- 内部根据 traits 选择对应的模板实例化内核
- 返回执行时间

**fmha_fwd_splitkv** / **fmha_fwd_appendkv**：
- 对应的 Split-KV 和 AppendKV 运行时接口

---

**ROCm 相关**：
- 完全基于 AMD HIP 平台
- 使用 Composable Kernel (CK) tile 库
- dim3 grids 用于 HIP 内核启动

**Backward 相关**：
- 此文件仅包含前向传播，无反向传播逻辑
