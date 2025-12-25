# MHA.cpp/h 核心功能分析

这两个文件实现了基于 cuDNN 的 Multi-Head Attention (MHA) 操作，专门用于加速 Scaled Dot-Product Attention (SDPA) 的前向计算。

## 主要组件

### 1. 三个核心函数接口 (MHA.h)

**run_cudnn_SDP_fprop** - 标准张量的前向传播
- 处理常规的 4D 张量 (batch, heads, seq_len, dim)
- 支持 Q、K、V 的注意力计算
- 可选的 attention bias、dropout、causal mask

**run_cudnn_SDP_fprop_nestedtensor** - 嵌套张量的前向传播  
- 处理变长序列的优化路径
- 使用 ragged offset 和 cumulative sequence lengths
- 输入是压缩的 3D 张量 (total_tokens, heads, dim)

**run_cudnn_SDP_bprop** - 反向传播
- 计算 dQ、dK、dV 梯度

### 2. 编译时特性检测 (MHA.cpp:5-32)

```cpp
#if defined(USE_ROCM) || !AT_CUDNN_ENABLED() || (CUDNN_VERSION < 8900)
```

如果不满足条件（ROCm、无 cuDNN、或 cuDNN < 8.9），所有函数抛出错误：
```
"PyTorch was not compiled with cuDNN Flash Attention enabled!"
```

### 3. 参数缓存系统 (148-302)

**MHAParams 结构体**
- 存储张量维度、步长、数据类型
- 设备 ID、dropout 概率、causal 标志等

**MHACacheKeyWrapper**
- 将参数打包成可哈希的 key

**MHAGraphCache**
- 线程局部缓存，存储已构建的 cuDNN execution graph
- 避免重复构建图（性能优化）

### 4. 图构建函数

**build_graph_and_tensors** (406-518) - 标准版本
1. 创建 cuDNN frontend Graph 对象
2. 设置数据类型：IO 用 HALF/BFLOAT16，中间计算用 FLOAT
3. 构建张量属性：
   - Q, K, V：从输入张量获取 dims/strides
   - attn_scale：标量，pass-by-value
   - seed/offset：dropout 随机数生成
   - bias：可选的 attention bias
4. 调用 `sdpa()` 创建 SDPA 操作
5. 验证、构建执行计划、检查硬件支持

**build_graph_and_tensors_nestedtensor** (520-713) - 嵌套张量版本
关键差异：
- 硬编码 BSHD 维度给 cuDNN，但实际数据是 THD 布局
- 使用 `INT_MAX` 作为 batch stride（表示 ragged）
- 设置 ragged offset 张量 (RAG_Q_OFF 等)
- 启用 padding_mask

### 5. 执行流程

**run_cudnn_SDP_fprop** (843-950)

```
1. 检测 GPU 型号，处理 Blackwell (10.0) 的 dropout bug
   → 转换 seed/offset 为 int64

2. 分配输出张量
   - o: 匹配 q 的内存布局 (alloc_with_matching_layout)
   - softmaxstats: [b, h, s_q] float tensor

3. 查缓存
   - 命中 → 复用 graph
   - 未命中 → 调用 build_graph_and_tensors

4. 填充 variant_pack (数据指针映射)
   {Q: q.data_ptr(), K: k.data_ptr(), ...}

5. 分配 workspace，执行图
   mha_graph->execute(handle, variant_pack, workspace_ptr)

6. 更新缓存
```

**run_cudnn_SDP_fprop_nestedtensor** (952-1063)

类似流程，但：
- 不使用缓存（每次重建图）
- 计算 ragged offsets：`cum_seqlen_q.mul(h_q * d_qk)`
- 计算实际序列长度：`diff(cum_seqlen_q)`

### 6. 辅助工具函数 (304-404)

**fixSizeOneDimStrideSDPA**
- 修复维度为 1 时的 stride（cuDNN 要求）

**alloc_with_matching_layout**
- 创建与 q 相同内存布局的输出张量
- 通过 argsort strides 确定填充顺序

**permute_to_matching_layout**
- 调整 grad_output 的步长匹配 output

**same_strides**
- 比较两个张量的 stride 模式（忽略 size=1 的维度）

## 技术要点

### cuDNN Frontend API
使用 `cudnn_frontend.h` 的图模式 API：
- `fe::graph::Graph`：构建计算图
- `fe::graph::SDPA_attributes`：配置 attention 参数
- `fe::HeurMode_t::A`：自动选择执行策略

### 内存优化
- 线程局部缓存避免跨线程竞争
- workspace 通过 CUDACachingAllocator 管理
- 输出张量复用输入的内存布局

### 硬件适配
- Blackwell GPU (SM 10.0) 的 dropout 类型转换
- cuDNN 9.5.1 之前的 stride 约束处理

---

## 其他特性（简要）

**ROCm 路径**
- 完全禁用，抛出错误

**Backward 支持**
- `build_graph_and_tensors_backward`：构建反向图
- `run_cudnn_SDP_bprop`：计算梯度
- 处理 cuDNN < 9.5.1 的 grad_output stride 限制
- 独立的反向缓存 `mhagraphbackwardcache`
