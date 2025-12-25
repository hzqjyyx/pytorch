### TensorAccessor.h 核心功能分析

**文件目的**：提供CPU和CUDA张量的高效索引访问接口，支持多维张量的strided内存布局。

---

**核心类型**：

1. **PtrTraits 类族**
   - `DefaultPtrTraits`：普通指针
   - `RestrictPtrTraits`：带`__restrict__`修饰的指针（CUDA优化）

2. **TensorAccessor**（一级）
   - 用于CPU张量和CUDA kernel内的即时索引
   - 通过递归模板特化支持多维访问：`accessor[i][j][k]`
   - 存储指针到size/stride数组的引用
   - N维版本递归调用`operator[]`返回N-1维accessor
   - 1维特化版本直接返回元素引用

3. **GenericPackedTensorAccessor**（二级）
   - 用于CUDA host端向kernel传递张量数据
   - 将sizes和strides**复制**到成员数组中（便于GPU传输）
   - Device端的`operator[]`返回TensorAccessor而非PackedTensorAccessor
   - 提供`transpose()`方法交换维度的size/stride（无数据移动）
   - 支持int64_t和int32_t两种index_t

4. **Deprecated Aliases**
   - `PackedTensorAccessor`：`GenericPackedTensorAccessor`的旧名称
   - `PackedTensorAccessor32/64`：32/64位索引版本的type alias

---

**设计特点**：

- 模板参数：数据类型T、维度N、指针特征PtrTraits、索引类型index_t
- 支持non-contiguous内存布局（通过stride数组）
- Host-Device分工：`C10_HOST`/`C10_DEVICE`/`C10_HOST_DEVICE`宏标注执行位置
- 边界检查：1维特化版提供`bounds_check_()`方法

---

**使用场景对应**：

- **TensorAccessor**：`tensor.accessor<T,N>()`获取，CPU计算或已在kernel内
- **GenericPackedTensorAccessor**：包装Tensor后传递到CUDA kernel的参数

---

**主要功能点**：

- 提供strided多维数组的统一访问抽象
- 支持CPU和CUDA的编译时分化
- 零拷贝的维度变换（转置）
- 灵活的指针类型控制（支持__restrict__优化）
- 兼容不同整数索引类型的类型转换
