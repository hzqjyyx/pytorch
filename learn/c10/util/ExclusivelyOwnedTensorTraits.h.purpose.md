## ExclusivelyOwnedTensorTraits 功能分析

这是一个模板结构体，为 `ExclusivelyOwned<T>` 容器提供 Tensor 类型的特性定义。主要用于管理独占所有权的 Tensor 对象的生命周期。

### 核心要素：

**类型定义**
- `repr_type`: Tensor 的表示类型
- `pointer_type`: Tensor 指针
- `const_pointer_type`: Const Tensor 指针

**关键方法**

- `nullRepr()`: 返回默认构造的空 Tensor
- `createInPlace()`: 通过完美转发创建新 Tensor 实例
- `moveToRepr()`: 将右值引用的 Tensor 转移所有权
- `destroyOwned()`: 销毁 Tensor 并释放其底层 TensorImpl
  - 验证 refcount 为 1（或 UndefinedTensorImpl 为 0）
  - 验证 weakcount 为 1（或 UndefinedTensorImpl 为 0）
  - 特殊处理 UndefinedTensorImpl 单例（不删除）
  - 调试模式下清零计数器
- `take()`: 移动语义提取 Tensor
- `getImpl()`: 获取 Tensor 指针（可重载 const 版本）

### 设计目的：

- **独占所有权保证**: 通过 refcount/weakcount 检查确保 Tensor 只被单一 ExclusivelyOwned 持有
- **内存安全**: 负责正确的构造、销毁和所有权转移
- **特殊处理**: 为 UndefinedTensorImpl 这个全局单例提供独特的生命周期管理

**关键特性**：
- 完美转发支持
- Debug 断言验证所有权状态
- 右值引用优化
- 单例对象特殊处理
