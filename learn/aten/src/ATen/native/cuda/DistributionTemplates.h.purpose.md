## 核心功能：CUDA 分布采样的通用模板框架

这个文件提供了在 CUDA 上实现各种概率分布随机采样的通用基础设施，通过模板化设计实现代码复用。

## 关键组件

### 1. 执行策略计算 (`calc_execution_policy`)
- **作用**：为分布采样核函数计算最优的 grid/block 配置
- **核心逻辑**：
  - 固定 block_size = 256，限制 grid_size ≤ 4
  - 根据 SM 数量和每个 SM 的最大 block 数动态调整 grid 大小
  - 计算 `counter_offset`：确保 Philox RNG 的偏移量足够大，避免随机数重复
    ```
    counter_offset = ceil(numel / (block_size × grid_size × unroll_factor)) × 4
    ```

### 2. Grid-stride Loop 核函数 (`distribution_elementwise_grid_stride_kernel`)
- **设计模式**：使用网格跨步循环而非一线程一元素，提升带宽利用率
- **工作流程**：
  1. 每个线程初始化独立的 `curandStatePhilox4_32_10_t` 状态
  2. 线程以 `blockDim.x × gridDim.x × unroll_factor` 为步长遍历元素
  3. 每次迭代调用 `dist_func` 生成随机数（如 `float4`）
  4. 通过 `unroll_factor` 展开循环，从 `float4` 中取出多个随机值
  5. 应用 `transform_func` 转换随机值并写入输出

### 3. 分发核函数 (`distribution_nullary_kernel`)
- **职责**：
  - 处理大张量的 32-bit 索引限制（递归分割子迭代器）
  - 管理 RNG 状态（加锁获取 `philox_cuda_state`）
  - 区分连续/非连续内存布局优化路径
    - **连续**：直接使用步长计算偏移
    - **非连续**：使用 `OffsetCalculator` 计算多维索引

### 4. 二元分布核函数 (`distribution_binary_elementwise_kernel`)
- 用于需要两个输入参数的分布（如条件采样）
- 采用寄存器分块优化：
  - 先将输入加载到寄存器数组 `inputs_1[thread_work_size()]`
  - 再计算并写回，减少全局内存访问

## 具体分布实现

所有分布都遵循相同模式：**curand 生成基础随机数 → transformation 函数转换**

### 均匀整数分布 (`random_from_to_kernel`)
- **小范围** (`range < 2^28`)：使用 `uint4`（4×32-bit）
- **大范围** (`range ≥ 2^28`)：使用 `ulonglong2`（2×64-bit），将两个 32-bit 拼接成 64-bit
- **特殊情况** (`random_full_64_bits_range_kernel`)：处理 `[int64_min, int64_max]` 全范围

### 连续分布（Normal/Uniform/LogNormal/Exponential/Cauchy）
- **共同点**：
  - 双精度用 `curand_uniform2_double` / `curand_normal2_double`（生成 `double2`）
  - 单精度用 `curand_uniform4` / `curand_normal4`（生成 `float4`）
- **差异化转换**：
  - Normal: `rand × std + mean`
  - LogNormal: `exp(normal_rand)`
  - Uniform: `rand × (to - from) + from`，并反转边界 `(0,1]` → `[0,1)`
  - Exponential: `-log(rand) / lambda`
  - Cauchy: `median + sigma × tan(π × (rand - 0.5))`

### 离散分布（Geometric/Bernoulli）
- **Geometric**：使用均匀分布通过 `transformation::geometric` 转换
- **Bernoulli**：
  - **标量概率**：通过 `rand ≤ p` 比较
  - **张量概率**：使用专门的 `bernoulli_tensor_cuda_kernel`，一次处理 4 个元素

## 设计亮点

1. **类型抽象**：
   - `scalar_t`：输出类型
   - `accscalar_t`：累加精度类型（如 Half → float）
   - `dist_func_return_t`：决定 `unroll_factor`（如 `float4` → 4）

2. **Lambda 闭包**：
   - `dist_func`：封装 curand 调用
   - `transform_func`：封装数学变换，捕获分布参数（mean/std/lambda 等）

3. **内存访问优化**：
   - 连续张量：`stride × idx` 直接计算
   - 非连续张量：`OffsetCalculator` 处理广播/转置

4. **RNG 线程安全**：通过 `std::lock_guard` 保护 generator 的 philox 状态获取

## ROCm 相关
- 可能存在 HIP 版本的条件编译（文件中有 `#ifdef FBCODE_CAFFE2` 分支）

## Backward 相关
- 此文件仅处理前向采样，采样操作本身不可导，无 backward 实现
