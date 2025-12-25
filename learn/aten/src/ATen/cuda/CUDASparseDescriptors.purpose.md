# CUDASparseDescriptors 核心功能

这两个文件提供了 **cuSPARSE 库描述符对象的 C++ RAII 封装**，用于管理 CUDA 稀疏矩阵运算中的各种描述符生命周期。

## 主要组件

### 1. 描述符基类模板 (CUDASparseDescriptors.h:15-36)

```cpp
template <typename T, cusparseStatus_t (*destructor)(T*)>
class CuSparseDescriptor
```

- 使用 `std::unique_ptr` + 自定义删除器实现自动资源管理
- 封装 cuSPARSE 原始指针类型，自动调用相应的销毁函数
- 避免手动调用 `cusparseDestroy*` 系列函数

### 2. 稠密矩阵描述符 (CuSparseDnMatDescriptor)

**构造过程** (CUDASparseDescriptors.cpp:59-114):

```cpp
cusparseDnMatDescr_t createRawDnMatDescriptor(const Tensor& input, 
                                               int64_t batch_offset, 
                                               bool is_const)
```

核心逻辑：
- **布局检测**: 判断输入张量是行主序还是列主序 (68-76 行)
- **Leading dimension**: 根据布局计算 `leading_dimension` (步长)
- **批处理支持**: 通过 `batch_offset` 参数支持批量矩阵
  - `batch_offset >= 0`: 处理单个批次，偏移指针到特定批次
  - `batch_offset == -1`: 设置 strided batch (107-112 行)
- **数据类型验证**: `check_supported_cuda_type()` 检查 GPU 计算能力是否支持特定数据类型
  - Float16 需要 compute capability >= 5.3
  - BFloat16 需要 compute capability >= 8.0

### 3. 稠密向量描述符 (CuSparseDnVecDescriptor)

**约束** (CUDASparseDescriptors.cpp:124-139):
- 仅支持 1D 张量或 2D 列向量 (`dim==2 && size(-1)==1`)
- 必须连续存储 (cuSPARSE 不支持非连续向量)

### 4. CSR 稀疏矩阵描述符 (CuSparseSpMatCsrDescriptor)

**核心功能** (CUDASparseDescriptors.cpp:141-216):

处理 CSR (Compressed Sparse Row) 格式的稀疏矩阵：
- **三个关键数组**:
  - `crow_indices`: 行偏移数组 (size = rows+1)
  - `col_indices`: 列索引数组 (size = nnz)
  - `values`: 非零元素值 (size = nnz)

- **批处理计算** (164-186 行):
  - 分别计算三个数组的批次步长
  - 使用指针算术偏移到特定批次: `data_ptr + batch_offset * stride * itemsize`

- **批处理模式** (193-213 行):
  - **广播模式**: 当索引/值是 1D 时，设置 stride=0 允许跨批次广播
  - **独立批次**: 当索引/值是 2D 时，每个批次有独立数据 (但 cuSPARSE 实现有问题，见 198-201 行注释)

### 5. 辅助功能

**索引类型转换** (CUDASparseDescriptors.cpp:48-57):
```cpp
cusparseIndexType_t getCuSparseIndexType(const c10::ScalarType& scalar_type)
```
- `Int` → `CUSPARSE_INDEX_32I`
- `Long` → `CUSPARSE_INDEX_64I`

**特殊方法**:
- `CuSparseSpMatCsrDescriptor::get_size()`: 获取矩阵维度和非零元素数 (h:204-212)
- `set_tensor()`: 更新描述符指向的数据指针 (h:214-227)
- `set_mat_fill_mode()` / `set_mat_diag_type()`: 设置三角矩阵属性 (h:230-248)

## 设计要点

1. **RAII 资源管理**: 所有描述符在析构时自动释放 cuSPARSE 资源
2. **批处理灵活性**: 支持单批次提取和多批次 strided 访问
3. **类型安全**: 编译时绑定正确的销毁函数
4. **平台兼容性**: 通过预处理宏适配不同 CUDA/cuSPARSE 版本

---

## ROCm/HIP 相关差异
- ROCm 仅支持列主序稠密矩阵 (h:81-83)
- 使用 `std::remove_pointer_t` 处理 HIP 类型定义 (h:64-74)

## Backward 兼容性处理
- CUDA 11 不支持 const 描述符，通过 `destroyConstDnMat()` 包装处理 (cpp:10-13, h:78)
- 条件编译区分 CUDA 12+ 的 const 描述符 API (h:38-61, 158-197)
