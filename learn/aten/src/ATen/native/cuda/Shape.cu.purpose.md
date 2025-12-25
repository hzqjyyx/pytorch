# Shape.cu 主要功能分析

这个文件实现了 CUDA 上的张量拼接（concatenation）操作，即 `torch.cat()` 的 CUDA 后端。

## 核心实现思路

**并行拼接策略**：将多个输入张量沿指定维度拼接到输出张量中，使用 GPU 并行处理多个输入张量。

## 关键组件

### 1. Grid 配置计算

```cpp
getCatGrid()           // 通用网格配置：基于 SM 数量，gridDim.y = nTensors
getCatGridContig()     // 连续张量优化：使用 128 线程块，向量化加载
```

- 每个 block.y 处理一个输入张量
- 动态调整线程块数量，避免小张量时过多线程空转

### 2. 索引转换

**`CatArrIndexToOffset`**：将线性索引转换为多维偏移
- 支持 Contiguous 和 ChannelsLast 内存格式
- 根据拼接维度调整索引计算

### 3. 三种 Kernel 实现

#### (1) **CatArrayBatchedCopy** (shape.cu:165)
通用版本，支持非连续张量
```cpp
// 根据每个张量的 isContiguous 标志选择路径
if (isContig) {
    output[dataOffset + elementOffset] = data[tid];
} else {
    // 需要计算输入张量的 elementOffset
}
```

#### (2) **CatArrayBatchedCopy_contig** (shape.cu:201)
连续张量专用，简化索引计算
```cpp
// 直接使用 tid 访问输入数据
output[dataOffset + elementOffset] = data[tid];
```

#### (3) **CatArrayBatchedCopy_aligned16_contig** (shape.cu:234)
**性能优化版本**，使用 128-bit 向量化加载
```cpp
constexpr int kILP = 16 / sizeof(T);  // 每次加载多个元素
// 使用 aligned_vector 一次性加载 kILP 个元素
((LT*)reg_data)[0] = const_cast<LT*>((LT*)(data + inputOffset))[0];
```

要求：
- 输入地址 16 字节对齐（`is_aligned_vec4()`）
- 数据类型 >= 4 字节（float, int32, double 等）
- 所有输入连续

### 4. Batching 机制

**`CAT_ARRAY_BATCH_SIZE = 128`**：每次最多处理 128 个输入张量
- 分批处理避免元数据过大
- 使用结构体 `CatArrInputTensorMetadata` 传递元数据到 kernel

### 5. 调度逻辑 (shape.cu:446 `cat_out_cuda`)

```
if (条件1: 所有输入连续 + 32位可索引 + dims≤4):
    parallel_cat<batch_size=128, stride_size=1>
    └─ 选择 aligned16/contig/通用 kernel

else if (条件2: 非连续但 dims≤4 + Contiguous格式):
    parallel_cat<batch_size=64, stride_size=64>  // 需要传递 stride 信息
    
else:
    fallback: 串行 narrow + copy
```

## 内存格式支持

- **Contiguous**：标准行优先布局
- **ChannelsLast/ChannelsLast3d**：NHWC 布局，通过重新排列维度映射到连续张量逻辑

```cpp
// NCHW -> NHWC 维度重映射
outputParam.tensorSize[nDims-1] = out.size(1);  // C 移到最后
```

## 性能优化点

1. **向量化加载**：16 字节对齐时使用 `int4` 加载
2. **Grid-stride loop**：`tid += gridDim.x * blockDim.x` 处理大张量
3. **分支最小化**：预检查连续性，选择最优 kernel
4. **类型擦除**：使用 `OpaqueType<N>` 避免相同大小类型重复编译

---

**次要内容**：
- ROCm 优化路径：不使用 aligned16 kernel，固定 256 线程块
- Backward 相关：无，这是纯前向操作
