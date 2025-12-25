## Utils.h 主要功能

**宏定义和工具函数**
- `AT_DISALLOW_COPY_AND_ASSIGN` 宏：禁用类的拷贝构造和赋值操作
- `_crash_if_asan()` 函数声明：用于 ASAN（AddressSanitizer）测试的故意缓冲区溢出

**张量验证和转换**
- `checked_dense_tensor_list_unwrap()` ：将 `ArrayRef<Tensor>` 转换为 `vector<TensorImpl*>`，并验证：
  - 张量布局必须是 Strided（密集）
  - 所有张量设备类型一致
  - 所有张量数据类型一致
  
- `check_intlist()` 模板函数：验证整数列表的长度和内容
  - 支持单值自动填充到指定长度
  - 验证列表大小是否匹配预期 N

**张量创建函数声明（detail 命名空间）**
- `tensor_cpu()` ：在 CPU 上从数组创建张量
- `tensor_backend()` ：在指定设备上创建张量（通过 CPU 中间转换）
- `tensor_complex_cpu()` ：在 CPU 上创建复数张量
- `tensor_complex_backend()` ：在指定设备上创建复数张量

---

## Utils.cpp 主要功能

**ASAN 调试函数实现**
- `_crash_if_asan()` ：通过访问数组越界位置来触发 ASAN 检测

**张量创建实现（detail 命名空间）**
- `tensor_cpu()` ：
  - 创建空张量
  - 使用 `AT_DISPATCH_ALL_TYPES_AND_COMPLEX` 宏调度所有标量类型
  - 复制值到张量数据指针

- `tensor_backend()` ：
  - 先在 CPU 创建张量
  - 再通过 `.to()` 转移到目标设备

- `tensor_complex_cpu()` ：
  - 针对复数类型的特化版本
  - 使用 `AT_DISPATCH_COMPLEX_TYPES` 只调度复数类型

- `tensor_complex_backend()` ：类似 `tensor_backend()` 的复数版本

**公共 API（通过宏生成）**
- 生成多个 `tensor()` 重载函数（用于不同标量类型）
- 根据设备类型自动选择 CPU 或后端实现

---

## 核心功能总结

- **张量验证**：检查张量列表的布局、设备、数据类型一致性
- **参数验证**：检查整数列表的长度和大小
- **张量创建工厂函数**：从数组值创建张量，支持 CPU 和非 CPU 设备
- **宏驱动代码生成**：为所有标量类型自动生成张量创建函数重载
- **设备抽象**：统一处理 CPU 和其他设备的张量创建

**主要特点：**
- 模板化设计，支持多种标量类型
- 宏驱动代码生成减少重复代码
- 类型和设备的早期验证
- CPU 作为中间层的跨设备张量创建策略
