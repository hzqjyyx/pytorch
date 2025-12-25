# TensorFactories 文件功能分析

## 核心职责

这两个文件实现了 PyTorch 中各种**张量创建和初始化操作**，提供了从零开始构造张量的基础设施。

## 主要功能模块

### 1. 空张量创建 (Empty Tensor Creation)

**empty 系列函数** (aten/src/ATen/native/TensorFactories.cpp:263-414)
- `empty_cpu`: 创建未初始化的张量
- `empty_strided_cpu`: 创建指定步长的张量
- `empty_permuted_symint`: 创建按物理布局排列的张量
- `empty_like`: 创建与给定张量相同形状的空张量
- 支持确定性填充模式：当启用确定性算法时，用 NaN（浮点型）或最大值（整数型）填充

**实现细节**:
```cpp
// 确定性填充逻辑 (TensorFactories.h:118-141)
if (tensor.is_floating_point()) {
    tensor.fill_(std::numeric_limits<scalar_t>::quiet_NaN());
} else {
    tensor.fill_(std::numeric_limits<scalar_t>::max());
}
```

### 2. 数值范围张量 (Range Tensors)

**arange** (aten/src/ATen/native/TensorFactories.cpp:136-177)
- 生成等差数列张量
- 三个重载版本：`arange(end)`, `arange(start, end)`, `arange(start, end, step)`
- 自动类型推断：如果所有参数都是整数，默认使用 `Long` 类型

**linspace/logspace** (aten/src/ATen/native/TensorFactories.cpp:804-960)
- `linspace`: 线性等分区间
- `logspace`: 对数等分区间
- 支持复数类型的自动推断

**range** (aten/src/ATen/native/TensorFactories.cpp:1435-1460)
- 已弃用的范围生成函数（建议使用 arange）

### 3. 常量填充张量 (Constant Fill)

**zeros** (aten/src/ATen/native/TensorFactories.cpp:1615-1760)
- 创建全零张量
- 支持稀疏压缩格式 (`sparse_csr`, `sparse_bsr` 等)
- `_efficientzerotensor`: 使用特殊的零张量分配器，不实际分配内存

**ones** (aten/src/ATen/native/TensorFactories.cpp:964-1004)
- 创建全一张量
- 内部调用 `full(size, 1.)`

**full** (aten/src/ATen/native/TensorFactories.cpp:714-773)
- 用指定标量值填充张量
- 自动类型推断：根据 fill_value 类型选择 dtype
  - Boolean → `kBool`
  - Integer → `kLong`
  - Complex → `ComplexFloat/ComplexDouble`
  - Float → 默认浮点类型

### 4. 随机张量生成 (Random Tensors)

**rand** (aten/src/ATen/native/TensorFactories.cpp:1047-1104)
- 生成 [0, 1) 均匀分布的随机张量
- 调用 `uniform_(0, 1, generator)`

**randn** (aten/src/ATen/native/TensorFactories.cpp:1235-1320)
- 生成标准正态分布 N(0, 1) 的随机张量
- `normal`: 生成指定均值和标准差的正态分布

**randint** (aten/src/ATen/native/TensorFactories.cpp:1108-1231)
- 生成整数随机张量
- 支持 `[0, high)` 或 `[low, high)` 区间

**randperm** (aten/src/ATen/native/TensorFactories.cpp:1371-1431)
- 生成随机排列 `[0, n-1]`
- CPU 实现使用 Fisher-Yates 洗牌算法
- 对于大数值（≥ 2^32/20）使用 64 位随机数避免偏差

### 5. 单位矩阵 (Identity Matrix)

**eye** (aten/src/ATen/native/TensorFactories.cpp:623-684)
- 创建单位矩阵（对角线为 1，其余为 0）
- CPU 实现使用并行化填充对角线元素
- 支持非方阵（通过 `n` 和 `m` 参数）

### 6. 复数张量构造 (Complex Tensors)

**complex** (aten/src/ATen/native/TensorFactories.cpp:222-240)
- 从实部和虚部张量构造复数张量
- 自动类型提升：`Float → ComplexFloat`, `Double → ComplexDouble`

**polar** (aten/src/ATen/native/TensorFactories.cpp:242-260)
- 从模和角度构造复数张量（极坐标形式）
- 使用 dispatch stub 机制委托具体实现

### 7. 窗函数 (Window Functions)

实现了多种信号处理窗函数：

**bartlett_window** (aten/src/ATen/native/TensorFactories.cpp:1764-1804)
- 巴特利特窗（三角窗）

**blackman_window** (aten/src/ATen/native/TensorFactories.cpp:1808-1848)
- 布莱克曼窗
- 公式：`0.42 - 0.5*cos(2πn) + 0.08*cos(4πn)`

**hamming_window** (aten/src/ATen/native/TensorFactories.cpp:1852-1930)
- 汉明窗
- 支持自定义 alpha/beta 参数

**hann_window** (aten/src/ATen/native/TensorFactories.cpp:1934-1966)
- 汉宁窗（实际是 alpha=0.5, beta=0.5 的汉明窗）

**kaiser_window** (aten/src/ATen/native/TensorFactories.cpp:1970-2037)
- 凯塞窗，支持自定义 beta 参数

### 8. 三角索引生成 (Triangular Indices)

**tril_indices/triu_indices** (aten/src/ATen/native/TensorFactories.cpp:1464-1565)
- 生成下三角/上三角矩阵的索引
- 返回形状为 `[2, num_elements]` 的张量
- CPU 实现直接计算索引，避免生成完整矩阵

**辅助函数** (TensorFactories.h:38-63)
```cpp
inline int64_t get_tril_size(int64_t row, int64_t col, int64_t offset) {
    // 计算梯形或矩形区域的元素数量
    // 处理两种情况：纯梯形 或 梯形+矩形
}
```

### 9. 其他工具函数

**scalar_tensor** (aten/src/ATen/native/TensorFactories.cpp:1008-1043)
- 从标量创建 0 维张量
- CPU 快速路径优化，跳过设备分发

**vander** (aten/src/ATen/native/TensorFactories.cpp:2041-2070)
- 生成 Vandermonde 矩阵
- 使用 `cumprod` 实现幂次计算

**from_file** (aten/src/ATen/native/TensorFactories.cpp:2096-2127)
- 从文件映射创建张量
- 使用 `MapAllocator` 实现内存映射

**clone** (aten/src/ATen/native/TensorFactories.cpp:2131-2154)
- 深拷贝张量
- `MemoryFormat::Preserve` 模式保留原始步长

## 关键设计模式

### 1. 两阶段构造模式
大多数函数采用 "先创建空张量，再填充" 的模式：
```cpp
auto result = at::empty(size, options);
return result.fill_(value);
```

### 2. TensorOptions 封装
所有函数使用 `TensorOptions` 统一管理 dtype/layout/device/pin_memory：
```cpp
TensorOptions options = TensorOptions()
    .dtype(dtype)
    .layout(layout)
    .device(device)
    .pinned_memory(pin_memory);
```

### 3. 类型推断机制
- `infer_full_options`: 根据填充值推断 dtype
- `linspace_logspace_infer_options`: 根据起止值推断复数/浮点类型

### 4. Dispatch Stub
复数操作使用分发存根实现跨设备：
```cpp
DEFINE_DISPATCH(complex_stub);
DECLARE_DISPATCH(binary_fn, complex_stub);
```

## TensorFactories.h 辅助组件

### ZeroTensorAllocator (TensorFactories.h:145-162)
特殊分配器，用于高效零张量：
- `allocate()` 总是返回 `nullptr`
- 不实际分配内存，依赖调度键 `ZeroTensor` 处理

### 精度检查 (TensorFactories.h:82-113)
`check_supported_max_int_with_precision` 确保数值在浮点表示范围内：
- Half: 最大 2049
- Float: 最大 2^24+1
- Double: 最大 2^53+1

### 三角矩阵计算 (TensorFactories.h:38-63)
`get_tril_size` 计算下三角元素数量，处理两种几何情况：
1. 纯梯形：`row + offset <= col`
2. 梯形+矩形：`row + offset > 0`

---

## 忽略内容概述

**ROCm 相关**:
- 无明显 ROCm 特定代码（通过设备分发机制统一处理）

**Backward 相关**:
- 这些是张量创建原语，不涉及梯度计算
- `_cast_*` 系列函数 (TensorFactories.cpp:421-435) 用于类型转换，可能在自动求导中作为叶子节点
