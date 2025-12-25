## QTensorImpl 主要功能

QTensorImpl 是 PyTorch 中用于量化张量（Quantized Tensors）的张量实现类。

### 核心职责

**继承关系**：继承自 `c10::TensorImpl`，为量化张量提供专门的实现。

**关键成员**：
- `quantizer_`：存储量化器指针，定义量化方案和参数
- 继承自父类的所有张量元数据（大小、步长、存储指针等）

### 构造函数

提供两个构造函数重载：
1. 基础版本：接收 `Storage`、`DispatchKeySet`、`TypeMeta` 和 `Quantizer`
2. 带 `ImplType` 版本：用于特定的实现类型初始化

### 主要方法

**量化器管理**：
- `quantizer()`：获取量化器
- `set_quantizer_()`：设置量化器

**浅拷贝操作**：
- `shallow_copy_and_detach()`：创建浅拷贝，支持版本计数和元数据变更控制（两个重载版本）
- `shallow_copy_from()`：从另一个 QTensorImpl 复制元数据

**元数据复制**：
- 静态方法 `copy_tensor_metadata()`：复制张量元数据字段（大小、步长等）和量化器信息

### 主要特点

- **bullet-point 总结**：
  - 为量化张量提供专门的 TensorImpl 实现
  - 管理量化参数通过 Quantizer 对象
  - 支持浅拷贝和元数据操作，保持量化信息的一致性
  - 集成版本计数和元数据变更控制机制
  - 提供标准 TensorImpl 接口（`tensorimpl_type_name()` 等）
