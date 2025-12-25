## Unique.cu 文件功能总览

这个文件实现了 **PyTorch 张量的唯一元素计算功能** (unique operations)，专门为 CUDA GPU 设备优化。以下是主要功能分解：

### **核心功能模块**

| 功能 | 说明 |
|------|------|
| **`compute_unique`** (40-86行) | 通用的唯一元素计算核心函数，使用 Thrust 库进行GPU并行处理 |
| **`unique_dim_cuda_template`** (89-182行) | 沿指定维度计算唯一元素 |
| **导出的 API 函数** (187-230行) | 暴露给 PyTorch 框架的接口 |

---

### **主要功能详解**

#### 1. **`compute_unique` 核心算法** (40-86行)
计算张量中的唯一元素，支持三个可选输出：

- **逆索引 (inverse_indices)**：原始张量中每个元素映射到唯一元素的索引
- **计数 (counts)**：每个唯一元素在原始张量中出现的次数
- **唯一元素个数 (num_out)**：去重后的元素总数

关键操作：
```
- adjacent_difference: 标记相邻不同的元素位置
- inclusive_scan: 扫描生成逆索引
- thrust::unique/unique_by_key: GPU并行去重
```

#### 2. **`unique_dim_cuda_template` 维度级别去重** (89-182行)
对多维张量沿特定维度去重：

- 将张量重塑为 2D (维度大小 × 其他维度)
- 自定义比较器在扁平化数据上逐元素比较
- 使用排序后的索引进行去重
- 最后用 `index_select` 恢复原始形状

#### 3. **导出 API 函数** (187-230行)

| 函数 | 功能 |
|------|------|
| `_unique_cuda` | 全局唯一元素，返回值和逆索引 |
| `_unique2_cuda` | 全局唯一元素，返回值、逆索引和计数 |
| `unique_dim_cuda` | 沿维度去重（不连续） |
| `unique_dim_consecutive_cuda` | 沿维度去重（连续） |
| `unique_consecutive_cuda` | 全局或沿维度的连续去重 |

---

### **技术要点**

1. **Thrust 库使用**：GPU并行算法库（thrust::sort, thrust::unique, thrust::scan等）
2. **流管理**：使用 `ThrustAllocator` 和当前CUDA流确保线程安全
3. **内存优化**：使用 `c10::load` 进行缓存友好的数据加载
4. **数据类型支持**：支持所有标准PyTorch数据类型（整数、浮点、布尔、bf16等）

这个文件是 PyTorch CUDA 后端中的关键组件，为张量操作提供高性能的GPU实现。
