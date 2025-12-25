## Convolution.cpp 核心功能

这是 PyTorch ATen 库中卷积操作的**统一调度入口**和**后端选择层**，负责将高级卷积 API 路由到最优的底层实现。

### 核心架构

**1. 后端选择系统 (Backend Selection)**
- `_select_conv_backend()`: 根据输入张量属性、参数和硬件选择最优后端 (1202行)
- `select_conv_backend()`: 公开的后端选择接口 (1311行)
- 支持 20+ 种后端实现:
  - **CuDNN**: GPU 标准卷积/转置卷积
  - **MIOpen**: AMD GPU 
  - **MKLDNN**: CPU 优化库
  - **XNNPACK**: 移动端优化
  - **Winograd3x3Depthwise**: CPU depthwise 推理优化
  - **Slow2d/3d**: CPU fallback 实现
  - **NnpackSpatial**: NNPACK 加速
  - **CudaDepthwise2d/3d**: 专门的 depthwise GPU 实现

**2. 卷积参数管理**
- `ConvParams<T>` 模板类 (284行): 封装所有卷积参数
  - stride, padding, dilation, groups
  - transposed, output_padding
  - benchmark, deterministic, cudnn_enabled, allow_tf32
- 智能参数检查: 负padding、非正stride、dilation验证等

**3. 前向计算入口**
- `_convolution()`: 主入口函数 (1468行)，完整流程:
  1. 参数扩展和验证
  2. 1D→2D 视图转换 (view4d/view3d)
  3. 后端选择 + 内存格式决策
  4. 巨型 switch-case 调度到具体实现 (1521-1688行)
  5. 特殊情况处理 (空张量、分组卷积等)

**4. 高级 API 封装**
- `conv1d/2d/3d_symint()`: 支持符号整数的用户接口 (915行起)
- `conv_transpose1d/2d/3d_symint()`: 转置卷积接口 (1116行起)
- `_convolution_mode_symint()`: 支持 "same"/"valid" padding 字符串 (1056行)
- `convolution_same()`: 实现 TensorFlow 风格的 same padding (991行)

**5. 复数卷积**
- `complex_convolution()`: 用高斯技巧实现复数卷积 (845行)
  - 将 `conv(W, x, b)` 分解为 3 次实数卷积
  - 公式: `a - b + i(c - a - b)`，其中 `a=conv(Wr,xr)`, `b=conv(Wi,xi)`, `c=conv(Wr+Wi, xr+xi)`

**6. 性能优化启发式**
- `check_cudnn_depthwise_workload()`: 检查是否启用 FP16 depthwise cuDNN 内核 (96行)
  - 基于 batch_size, channels, width, stride 的复杂决策树
  - cuDNN 8.2+ 版本使用简化版 `check_cudnn_depthwise_workload_with_filter()`
- 内存格式选择: `determine_backend_memory_format()` (1412行)
  - 根据后端自动选择 Contiguous/ChannelsLast/ChannelsLast3d

**7. 分组卷积处理**
- 对不支持原生分组的后端 (Slow2d等)，手动拆分 (1654-1661行):
  - 按 group 切分 input/weight/bias
  - 逐组计算后拼接结果

**8. 形状验证**
- `check_shape_forward()`: 验证输入/权重/bias 的形状兼容性 (656行)
- `batchify()`: 自动添加 batch 维度处理 (756行)
- 计算输出尺寸: `calc_output_size()` (1396行)

**9. 边界情况处理**
- **空张量**: 特殊路径避免进入后端 (1548-1568行)
- **零 batch/channel**: 直接返回空输出
- **64位索引**: cuDNN 9.3+ 或 V8 API 支持超大张量 (427行)

### 关键设计模式

- **策略模式**: 后端枚举 + switch 分发
- **模板编程**: `ConvParams<T>` 支持动态形状 (SymInt) 和静态形状 (int64_t)
- **渐进降级**: cuDNN → MIOpen → MKLDNN → CPU Slow → Overrideable
- **延迟计算**: 在参数检查后才选择后端，避免无效计算

---

### 简要总结 (忽略内容)

**ROCm 相关**:
- MiopenDepthwise/Miopen/MiopenTranspose 后端
- `use_miopen()` 检查和调用 (504, 1570-1585行)
- `miopen_convolution_backward_stub` 等分发器

**Backward 相关**:
- `_convolution_backward_nogroup_backend()`: 反向传播调度 (1924行)
- `convolution_backward_overrideable()`: 可重载反向接口 (1710行)
- `_convolution_double_backward()`: 二阶导数 (1727行)
- 多个 backward stub 注册 (615-636行)
