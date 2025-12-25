## EmptyTensor.cpp/h 主要功能

这两个文件负责 PyTorch 中空张量（empty tensor）的创建和存储大小计算。

### 核心功能

**1. 存储大小计算**

提供了计算张量底层存储所需字节数的函数：

- `computeStorageNbytesContiguous()` - 连续存储：计算公式为 `(元素总数 + storage_offset) * itemsize`
- `computeStorageNbytes()` - 非连续存储：根据 strides 计算最后一个元素的偏移量，公式为 `(storage_offset + 1 + Σ(stride[i] * (size[i] - 1))) * itemsize`

两者都有普通版本和 SymInt 版本（用于符号形状）。非移动版本包含溢出检查，限制在 `min(INT64_MAX, SIZE_MAX)`。

**2. 空张量创建**

通用模板函数 `_empty_generic()` 的工作流程：
- 检查尺寸非负（`check_size_nonnegative`）
- 对 ComplexHalf 类型发出实验性警告
- 计算所需存储字节数
- 创建 StorageImpl（可调整大小）
- 创建 TensorImpl 并设置连续尺寸
- 可选地根据 MemoryFormat 重新调整步幅

`_empty_strided_generic()` 类似，但直接设置自定义的 size 和 stride。

**3. CPU 张量创建**

- `empty_cpu()` - 创建 CPU 张量，支持 pinned memory
  - 通过 `GetCPUAllocatorMaybePinned()` 选择分配器
  - pin_memory 时优先使用 CUDA（向后兼容），其次 PrivateUse1
- `empty_strided_cpu()` - 创建带自定义步幅的 CPU 张量

提供了多个重载版本，接受 TensorOptions 或单独的参数。

**4. Meta 张量创建**

- `MetaAllocator` - 特殊分配器，总是返回 nullptr（不实际分配内存）
  - 用于形状推断和元编程，无需实际内存
- `empty_meta()` / `empty_strided_meta()` - 创建 meta 设备上的张量
  - 支持 SymInt 版本用于符号形状追踪
  - 目前不支持非 Strided layout

### 关键设计点

- **条件编译**：移动版本（`C10_MOBILE`）跳过溢出检查以减少开销
- **统一接口**：通过模板支持 `IntArrayRef` 和 `SymIntArrayRef`
- **设备无关**：通过 DispatchKeySet 和 Allocator 参数化设备类型
- **优化路径**：对默认尺寸 `[0]` 的张量跳过尺寸设置（除了 Meta）

---

**ROCm/HIP 相关**：
- `GetCPUAllocatorMaybePinned` 中提到 PrivateUse1 支持（可用于 ROCm）
- tunable/GemmHipblaslt.h, tunable/GemmRocblas.h 涉及 ROCm GEMM 优化

**Backward 相关**：
- 该文件声明 `TORCH_ASSERT_NO_OPERATORS`，不包含自动微分逻辑
- 仅负责前向传播的张量内存分配
