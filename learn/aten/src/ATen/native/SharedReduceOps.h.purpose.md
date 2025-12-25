# SharedReduceOps.h 核心功能分析

这是一个跨CPU/GPU的通用reduction操作模板库，定义了各种归约算子（reduction operators）的统一接口和实现。

## 核心设计模式

每个Ops结构体都实现了标准的归约接口：
- `reduce(acc, data, idx)` - 将单个元素归约到累加器
- `combine(a, b)` - 合并两个累加器
- `project(acc)` - 最终投影/转换结果
- `translate_idx(acc, base_idx)` - 索引转换
- `warp_shfl_down(acc, offset)` - GPU warp内通信（CUDA/HIP）

## 主要Ops实现

### 1. **WelfordOps** (lines 96-160)
Welford在线算法计算均值和方差：
```cpp
WelfordData { mean, m2, n, nf }
```
- `reduce`: 增量更新均值和M2统计量
- `combine`: 合并两个Welford状态（并行归约用）
- `project`: 根据correction和take_sqrt标志输出方差或标准差
- 用于`var`, `std`等操作

### 2. **MeanOps** (lines 162-190)
简单均值计算：
- `reduce/combine`: 累加求和
- `project`: 乘以factor（通常是1/N）

### 3. **范数Ops系列**

**NormZeroOps** (lines 289-313) - L0范数：
- 统计非零元素个数

**NormOneOps** (lines 319-342) - L1范数：
- `reduce`: 累加绝对值

**NormTwoOps** (lines 367-391) - L2范数：
- `reduce`: 累加平方（`abs_if_complex`处理复数）
- `project`: 开平方根

**NormOps** (lines 255-283) - Lp范数：
- `reduce`: 累加`|x|^p`
- `project`: 开p次方根

### 4. **绝对值极值Ops**

**AbsMinOps** (lines 196-220)：
- 最小绝对值

**AbsMaxOps** (lines 226-249)：
- 最大绝对值

### 5. **NanSumOps** (lines 393-416)
忽略NaN的求和：
```cpp
reduce: a + (isnan(b) ? 0 : b)
```

### 6. **Min/Max/Arg系列**

**MinMaxReductionOps** (lines 448-476)：
- 基础模板，存储`pair<scalar_t, index_t>`
- 比较器`comp_t`可以是`LessOrNan`或`GreaterOrNan`
- NaN优先；相等时选择较小索引

**ArgMinOps/ArgMaxOps** (lines 491-508)：
- 继承自`ArgReductionOps`
- `project`返回索引而非值-索引对

**MinOps/MaxOps** (lines 501-509)：
- `project`返回完整的值-索引对

**MinMaxOps** (lines 511-540)：
- 同时计算最小值和最大值
- 累加器类型：`pair<min_val, max_val>`

## 平台适配机制

### 宏定义策略：
```cpp
// GPU (CUDA/HIP)
#define MAX(X,Y) max_propagate_nan(X,Y)  // NaN传播
#define MIN(X,Y) min_propagate_nan(X,Y)
#define compat_pow c10::cuda::compat::pow

// CPU
#define MAX(X,Y) max_impl(X,Y)
#define compat_pow std::pow
```

### 数据结构：
- GPU用`thrust::pair`
- CPU用`std::pair`

### GPU特定优化：
所有Ops都实现了`warp_shfl_down`用于warp级别归约，使用`WARP_SHFL_DOWN`宏

## 复数支持

`abs_if_complex` (lines 348-361)：
- 普通类型：直接返回
- `std::complex/c10::complex`：取绝对值
- 通过模板特化和`AbsSwitch`标签分发

---

**ROCm相关**: HIP特殊处理NaN比较bug (issues/2209)，在`max/min_propagate_nan`中与CUDA实现不同

**Backward相关**: 文件本身不涉及backward，但这些Ops是reduction kernels的基础，被`TensorIterator`和各种reduction操作调用
