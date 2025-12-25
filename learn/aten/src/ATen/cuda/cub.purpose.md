## 文件功能分析

这两个文件是 PyTorch ATen 库中 CUB (CUDA Unbound) 库的包装层，提供 GPU 加速的扫描和排序操作。

**cub.h** 是头文件，声明了以下接口：

- `radix_sort_pairs()` / `radix_sort_keys()` - 基数排序，支持按 key 排序或同时排序 key-value 对
- `inclusive_sum_truncating()` / `inclusive_sum()` - 包含性扫描求和（前向累加）
- `exclusive_sum_in_common_type()` / `exclusive_sum()` - 排他性扫描求和（后向累加）
- `mask_exclusive_sum()` - 基于掩码的排他性扫描求和

**cub.cu** 是实现文件，提供了具体的 CUDA 核函数实现：

- 定义了 `SumOp` 和 `CountMaskOp` 仿函数用于扫描操作
- 实现了各扫描函数的模板特例化
- 在 `mask_exclusive_sum()` 中使用 CUB 的 `TransformInputIterator` 将 uint8_t 掩码转换为布尔值进行计数

**设计特点**：
- 使用"意图声明但未定义"的模板模式，避免每个编译单元都重新实例化
- 通过 `OpaqueType` 封装使值类型对排序算法透明，减少模板实例化数量
- 显式实例化特定类型组合（int32_t、int64_t），需要时在 cub.cu 中添加新类型

---

• **radix_sort_pairs/keys** - GPU 基数排序（key 或 key-value 对）
• **inclusive/exclusive_sum** - 前向/后向累加扫描
• **mask_exclusive_sum** - 带掩码的排他性求和扫描
• **模板显式特例化** - 只在 cub.cu 中实例化，减少编译体积
• **CUB 库包装** - 对 NVIDIA CUB 库的 C++ 接口封装
