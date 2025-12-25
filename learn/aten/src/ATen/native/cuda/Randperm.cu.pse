现在我已经了解了完整实现。根据规范，这里是 Randperm.cu 的 TorchPseudo DSL 伪代码：

```
--------------------------------
randperm_out_cuda
来源: aten/src/ATen/native/cuda/Randperm.cu
--------------------------------

【签名】
    randperm_out_cuda(n: int64, generator: Generator?, result: Tensor[n]) -> Tensor[n, int64]

    约束:
    - n >= 0
    - result 是输出张量，尺寸将调整为 [n]

【算法】
    # [Algorithm of randperm]
    # randperm 通过对随机生成的 key 进行排序来实现。
    # 当所有 key 互不相同时，所有排列出现的概率相等。
    #
    # 问题：两个 key 可能相同，会导致某些排列概率不均匀。
    # 解决：(1) 选择足够多的比特数以保证 key 重复概率在阈值以下
    #       (2) 排序后，额外的 kernel 对 key 值相同的"孤岛"进行洗牌

    if n == 0:
        return empty([], dtype=int64)

    # Step 1: 计算所需比特数
    # q = 0.9 为重复概率阈值
    # bits = ceil(log2(n - (6*n² + 1) / (12*log(q))))
    log_threshold_12 = log(0.9) * 12
    nd = float(n)
    bits = min(64, ceil(log2(nd - (6*nd*nd + 1) / log_threshold_12)))

    # Step 2: 生成 arange [0, 1, ..., n-1]
    range = arange(n, device=result.device, dtype=result.dtype)

    # Step 3: 准备输出缓冲
    # 如果 result 连续，直接使用；否则分配临时张量
    if result.is_contiguous():
        shuffled_data = result.data_ptr()
    else:
        shuffled = empty(n, device=result.device, dtype=result.dtype)
        shuffled_data = shuffled.data_ptr()

    # Step 4: 生成随机 key 并排序
    if bits <= 32:
        # 32-bit 随机数
        keys = empty(n, device=result.device, dtype=int32)
                     .random_(int32_min, int32_max, generator=generator)
        keys_tmp = empty_like(keys)
        keys_out = keys_tmp.mutable_data_ptr()

        # radix_sort_pairs: 按 key 排序，将对应的 range 值移动到 shuffled_data
        # 输入：keys[i] -> 待排 key，range[i] -> 对应的值
        # 输出：keys_out[i] -> 排序后的 key，shuffled_data[i] -> 排序后的值
        radix_sort_pairs(
            keys_in=keys,
            keys_out=keys_out,
            values_in=range,
            values_out=shuffled_data,
            count=n,
            descending=false,
            begin_bit=0,
            end_bit=bits
        )

        # Step 5: 处理 key 重复的孤岛
        # kernel 在孤岛内使用 Fisher-Yates 算法额外洗牌
        randperm_handle_duplicate_keys(
            keys=keys_out,
            data=shuffled_data,
            bits=bits,
            n=n,
            generator=generator
        )
    else:
        # 64-bit 随机数
        keys = empty(n, device=result.device, dtype=int64)
                     .random_(int64_min, int64_max, generator=generator)
        keys_tmp = empty_like(keys)
        keys_out = keys_tmp.mutable_data_ptr()

        radix_sort_pairs(
            keys_in=keys,
            keys_out=keys_out,
            values_in=range,
            values_out=shuffled_data,
            count=n,
            descending=false,
            begin_bit=0,
            end_bit=bits
        )

        randperm_handle_duplicate_keys(
            keys=keys_out,
            data=shuffled_data,
            bits=bits,
            n=n,
            generator=generator
        )

    # Step 6: 如果原始输出非连续，复制回去
    if not result.is_contiguous():
        result.copy_(shuffled)

    return result

【并行模式】
    radix_sort_pairs: GPU 基数排序，CUB 库实现
    randperm_handle_duplicate_keys: 自定义 kernel

    grid: ceil(n / 512)
    block: 512

    - 每个线程检查一个索引是否为孤岛的起点
    - 孤岛起点的线程计算孤岛大小
    - 孤岛起点的线程初始化独立 RNG，用 Fisher-Yates 算法洗牌孤岛内的值

【备注】
    参考文献:
    [1] https://osf.io/af2hy/
    - Birthday Paradox 分析：两个随机数相同的概率与 key 空间的关系
    - 比特数选择可保证重复概率 < 0.1

---

randperm_handle_duplicate_keys_kernel (并行部分)
来源: aten/src/ATen/native/cuda/Randperm.cuh
---

【签名】
    randperm_handle_duplicate_keys_kernel(
        keys: T*,           # 排序后的 key 数组，T 为 int32/int64
        data: scalar_t*,    # 对应的排列值数组
        mask: T,            # 掩码，只看低 bits 位：mask = (1 << bits) - 1
        n: int,
        philox_args: PhiloxCudaState  # RNG 参数
    )

【算法】
    tid = threadIdx.x + blockDim.x * blockIdx.x

    if tid >= n - 1:
        return  # 线程越界

    if (keys[tid] & mask) != (keys[tid + 1] & mask):
        return  # 不在孤岛中

    if tid != 0 and (keys[tid] & mask) == (keys[tid - 1] & mask):
        return  # 不是孤岛的起点

    # 现在这个线程是孤岛的起点

    # Step 1: 找出孤岛的大小
    island_size = 0
    while (tid + island_size < n) and (keys[tid + island_size] & mask) == (keys[tid] & mask):
        island_size += 1

    # Step 2: 初始化 RNG
    data_ptr = data + tid
    seed, offset = unpack_philox_state(philox_args)
    rng_state = CurandState()
    curand_init(seed, tid, offset, rng_state)

    # Step 3: 孤岛内 Fisher-Yates 洗牌
    for i in range(island_size - 1, 0, -1):
        r = curand(rng_state) % (i + 1)
        if i != r:
            swap(data_ptr[i], data_ptr[r])

【并行模式】
    warp-independent:
    - 不同的孤岛在不同的线程中独立处理
    - 每个孤岛起点线程独立运行 Fisher-Yates 算法

【备注】
    - 孤岛大小通常很小（通常只有 2-3 个元素）
    - 单线程顺序执行 Fisher-Yates 即可
    - 使用 Philox RNG 保证可重复性
```

---

## 核心要点

1. **算法思想**：
   - 基数排序 + 重复 key 处理
   - 先用随机 key 排序 `[0,1,...,n-1]`，得到随机排列
   - 若 key 重复，用额外 kernel 对孤岛内部进行洗牌

2. **比特数计算**：
   - Birthday Paradox：`bits = ceil(log2(n - (6n²+1)/(12log(0.9))))`
   - 保证 key 重复的概率 < 10%

3. **并行策略**：
   - CUB 基数排序（GPU 库）
   - 孤岛处理 kernel：每个孤岛起点线程独立运行 Fisher-Yates
