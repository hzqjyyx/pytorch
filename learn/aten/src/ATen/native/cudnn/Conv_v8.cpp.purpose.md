这个文件实现了 PyTorch 中使用 cuDNN v8 API 执行卷积操作的核心功能。

## 核心架构

文件采用两阶段策略来选择和执行卷积算法：

**1. GET 模式（非 benchmark）**
- 通过启发式方法（heuristics）快速获取引擎配置
- 尝试配置列表，找到第一个可执行的就停止
- 如果启发式失败，回退到 fallback 配置列表
- 优先级：heuristic configs → fallback configs

**2. FIND 模式（benchmark）**
- 生成所有可能的执行计划
- 实际运行并计时排序，选择最快的
- 结果缓存到 LRU cache 中
- 完成后可选择清空 CUDA 缓存（因为 benchmark 消耗大量内存）

## 关键组件

**Tensor Descriptor 构建** (`getTensorDescriptor`, `getTensorDescriptorWithTypeVirtual`)
- 将 PyTorch Tensor 转换为 cuDNN frontend tensor 描述符
- 处理内存格式（ChannelsLast/ChannelsLast3d）
- 计算对齐方式（alignment，1-32 字节）
- 支持虚拟 tensor（用于融合操作的中间结果）

**Operation Graph 构建**
- `build_opgraph`: 单个卷积操作图
- `build_opgraph_fused`: 融合操作图（Conv + Add + Bias + ReLU）
  - 使用虚拟中间 tensor 避免实际内存分配
  - 所有中间计算用 FLOAT 精度，最终输出转回原始类型

**缓存机制** (`BenchmarkCache`)
- 线程局部存储（thread_local），因为 cuDNN ExecutionPlan 不保证跨线程安全
- LRU 淘汰策略，默认限制 10000 个条目（约 2GiB）
- 缓存键包含：卷积参数、操作类型、各 tensor 的对齐方式
- 通过环境变量 `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT` 控制

**引擎配置过滤** (`filterEngineConfigs`)
- 过滤掉确定性要求下的非确定性引擎
- 过滤掉会降精度的配置（`DOWN_CONVERT_INPUTS`）
- Float32 且禁用 TF32 时，过滤掉 Tensor Core 引擎

**执行计划生成**
- `get_configs_from_heuristics`: 从启发式/fallback 获取配置
- `get_plans_from_find`: 生成所有计划，过滤工作空间超限的，然后计时排序
- 工作空间管理：查询可用 CUDA 内存块大小，逐步减半重试直到分配成功

**计划执行** (`run_conv_plan`, `run_conv_plan_fused`)
- 分配工作空间
- 根据操作类型设置正确的数据指针（forward/backward_data/backward_filter）
- 构建 VariantPack 并调用 `cudnnBackendExecute`

**错误处理**
- Plan errata filter：从 JSON 配置加载已知问题引擎列表并跳过
- 多层 try-catch：处理 cuDNN 异常、CUDA 错误、OOM
- 工作空间 OOM 时自动减半重试

**对外接口**
- `raw_cudnn_convolution_forward_out`: 前向卷积
- `raw_cudnn_convolution_add_relu_out`: 融合操作（Conv+Add+Bias+ReLU）

所有接口都检查 `cudnnv8_enabled_check_debug()`，如果禁用则回退到 v7 实现。

## 特殊处理

**Ubuntu 22+ 兼容性**（71-88 行）
- 预加载 `libcudnn_cnn_infer.so.8` 避免缺少 `libnvrtc.so` 时崩溃

**尺寸为 1 的维度**（95-96 行）
- 调用 `fixSizeOneDimStride` 修正 stride，确保 ChannelsLast 格式正确

**数据类型处理**
- BFloat16/Half 的卷积描述符强制用 FLOAT 计算类型
- 融合操作所有中间计算用 FLOAT 精度

---

**ROCm 相关**: 无（此文件仅 cuDNN，ROCm 使用 MIOpen）

**Backward 相关**:
- `CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR`: 计算输入梯度
- `CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR`: 计算权重梯度
- `raw_cudnn_convolution_backward_input_out`: 输入梯度接口
- `raw_cudnn_convolution_backward_weight_out`: 权重梯度接口
