# Histogram.cpp/h 核心功能

## 整体架构

这两个文件实现了 PyTorch 的多维直方图计算功能，对标 NumPy 的 `histogramdd`。核心是将输入数据分配到多维网格的 bins 中，统计每个 bin 的计数或加权和。

## 主要功能模块

### 1. 接口层次

提供三个层次的直方图函数：

- **histogramdd**: N维直方图，输入 shape 为 (M, N)，表示 M 个 N 维坐标点
- **histogram**: 1维直方图的便捷接口，内部将输入 reshape 成 (numel, 1) 后调用 histogramdd
- **histc**: 遗留接口，功能受限的1维直方图（torch.histc）

### 2. Bins 定义方式

支持两种方式指定 bins：

**方式一：TensorList bins**
- 每个维度提供一个1维 tensor，显式定义 bin edges
- 例如：bins[0] = [0, 1, 2, 5] 表示第0维有3个bins：[0,1), [1,2), [2,5]

**方式二：IntArrayRef bin_ct**
- 每个维度提供一个整数，表示等宽 bins 的数量
- 自动计算 bin edges：通过 `select_outer_bin_edges` 确定范围，用 `linspace` 生成等间距边界

### 3. 核心计算流程

```
输入验证 (histogramdd_check_inputs)
   ↓
准备输出 (histogramdd_prepare_out) - 根据 bin_ct resize hist 和 bin_edges tensors
   ↓
确定 bin edges:
   - 如果是 TensorList: 直接复制
   - 如果是 IntArrayRef: 先调用 select_outer_bin_edges 确定范围 → linspace 生成
   ↓
调用底层 stub 执行实际计算:
   - histogramdd_stub: 处理任意 bin edges
   - histogramdd_linear_stub: 优化的等宽 bins 路径
```

### 4. Bin Range 确定逻辑 (select_outer_bin_edges)

处理三种情况：

1. **显式指定 range**: 直接使用用户提供的 `[min0, max0, min1, max1, ...]`
2. **空输入**: 默认 [0, 1] 范围
3. **自动推断**: 调用 `histogram_select_outer_bin_edges_stub` (设备相关实现) 计算每维的 min/max

特殊处理：
- 如果 min == max，扩展为 [min-0.5, max+0.5] 避免除零
- 验证边界是有限值且 min ≤ max

### 5. Weight 和 Density 支持

**Weight tensor**:
- 可选参数，shape 必须匹配 input 去掉最后一维
- 每个坐标点贡献其对应的 weight 值到 bin（而非默认的1）

**Density 模式**:
- `density=false`: hist 包含每个 bin 的总计数/权重
- `density=true`: 归一化为概率密度（除以总和再除以 bin 体积）

### 6. 输入验证 (histogramdd_check_inputs)

检查约束：
- input 至少2维
- bins 数量 == input 最后一维 N
- 所有 bins tensor 都是1维且非空
- input/bins/weight 的 dtype 必须一致
- weight shape 必须匹配 input 去掉最后一维

### 7. Dispatch 机制

通过 `DEFINE_DISPATCH` 定义设备分发点：

```cpp
histogramdd_stub                    // 任意 bin edges
histogramdd_linear_stub             // 等宽 bins（可优化）
histogram_select_outer_bin_edges_stub  // 计算 min/max
```

实际计算逻辑在 CPU/CUDA 后端实现（本文件不包含）。

### 8. histc 特殊逻辑

`histc_select_outer_bin_edges` 的差异：
- 如果 min == max 且输入非空，自动用 aminmax 计算实际范围
- 扩展空范围时用 ±1（而非 ±0.5）
- 要求 min < max（严格小于，而非 ≤）

### 9. 公共接口函数

**返回 (hist, bin_edges) 的函数**:
```cpp
histogramdd(input, TensorList bins, ...)     // bins 已定义
histogramdd(input, IntArrayRef bin_ct, ...)  // 自动生成 bins
histogramdd(input, int64_t bins, ...)        // 所有维度用相同数量
```

**内部使用的下划线函数**:
```cpp
_histogramdd(...)              // 仅返回 hist
_histogramdd_bin_edges(...)    // 仅返回 bin_edges
```

## 关键设计点

1. **统一接口**: histogram 复用 histogramdd，通过 reshape 实现
2. **双路径优化**: 等宽 bins 用 linear_stub，任意 bins 用通用 stub
3. **自动范围推断**: 空 range 时自动调用 aminmax 确定合理边界
4. **边界情况处理**: 空输入、空范围、min==max 都有明确的默认行为
5. **类型安全**: 严格要求所有 tensor 的 dtype 一致

---

## 补充说明

**ROCm 相关**: 文件中无 ROCm 特定代码，设备分发统一处理  
**Backward 相关**: 无反向传播实现，histogram 是非可微操作
