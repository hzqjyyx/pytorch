这个文件是 PyTorch 对 NVIDIA CUB (CUDA Unbound) 库的封装层，提供了高性能并行原语的统一接口。

## 核心功能

### 1. CUB 命名空间隔离 (lines 12-31)
通过宏定义将 CUB 库封装在 `at_cuda_detail` 命名空间中，避免与其他使用 CUB 的库冲突：
```cpp
#define CUB_NS_PREFIX namespace at_cuda_detail {
#define CUB_NS_QUALIFIER ::at_cuda_detail::cub
```

### 2. 临时存储管理宏 (lines 38-45)
`CUB_WRAPPER` 宏自动处理 CUB 算法所需的临时存储：
- 第一次调用获取所需内存大小
- 通过 CUDA Caching Allocator 分配内存
- 第二次调用执行实际操作
- 自动检查 CUDA 错误

### 3. BFloat16 支持 (lines 55-85, 106-120)
为旧版本 CUB 提供 `c10::BFloat16` 的特化：
- `FpLimits<c10::BFloat16>`: 定义最大/最小值
- `NumericTraits<c10::BFloat16>`: 类型特征定义
- 根据平台映射到 `__nv_bfloat16` 或 `hip_bfloat16`

### 4. 分段排序 (lines 124-160)
`segmented_sort_pairs()` 封装 CUB 的分段基数排序：
- 支持升序/降序
- 支持自定义比特范围
- 自动处理 Half/BFloat16 类型转换
- 限制：元素数和分段数不超过 INT_MAX

### 5. Scan 操作（前缀和）

#### Inclusive Scan (lines 228-293)
- 对于大于 `max_cub_size` (2^30) 的数据分块处理
- 使用 `transform_vals` 内核在块间传递累积值
- 根据 CUB 版本选择 `FutureValue` 或自定义迭代器连接块

#### Exclusive Scan (lines 505-562)
类似 inclusive scan，但输出向右偏移一位，首元素为 init_value

#### Deterministic Scan (lines 479-501)
提供确定性结果的 scan 实现：
- 将数据分配到固定数量的 CTA（每个 SM 一个）
- 先计算每个块的部分和 (`calc_block_sums`)
- 再执行最终 scan (`final_scan_kernel`)
- 使用 `BlockPrefixCallbackOp` 在块间传递前缀

#### Scan by Key (lines 564-592)
按键值分组的 scan 操作，相同键值内累积，键值变化时重置

### 6. 其他并行原语

**unique_by_key** (lines 162-178): 按键值去重，保留每个键的第一个值

**unique** (lines 594-601): 去除连续重复元素

**run_length_encode** (lines 603-613): 游程编码，输出唯一值、计数和总长度

**reduce** (lines 615-624): 规约操作，应用二元操作符将序列归约为单个值

### 7. 性能优化

**分块处理策略** (lines 220-223): 
- CUB 理论支持 INT_MAX 元素但实际不稳定
- 在 INT_MAX/2 处分割大张量

**内存管理**:
- 使用 CUDA Caching Allocator 避免频繁分配
- 临时缓冲区自动释放

**类型转换** (lines 97-122):
- `cuda_type` 模板将 PyTorch 类型映射到 CUDA 原生类型
- 避免类型不匹配导致的性能损失

---

**ROCm 相关**: 使用 `hipcub` 替代 `cub`，宏 `ROCM_HIPCUB` 处理命名空间差异

**向后兼容**: 
- 旧版 CUB 缺少 `FutureValue` 时使用自定义 `chained_iterator`
- 手动实现 BFloat16 特化
