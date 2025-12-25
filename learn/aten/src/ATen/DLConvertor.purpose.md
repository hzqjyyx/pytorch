## DLConvertor 文件主要功能

**DLConvertor** 是 PyTorch ATen 与 DLPack 标准之间的桥接模块，实现 Tensor 对象与 DLPack 张量的相互转换。

### 核心函数

**getDLDataType()** (lines 7-94)
- 将 PyTorch Tensor 的标量类型转换为 DLPack 数据类型
- 处理整数、浮点数、复数、布尔值等多种类型
- 对不支持的类型（float8、量化类型、bit types）抛出错误

**getDLDevice()** (lines 96-133)
- 将 PyTorch 设备类型映射到 DLPack 设备类型
- 支持 CPU、CUDA/HIP/ROCM、OpenCL、XPU、MAIA、PrivateUse1 等后端
- 在 ROCm 编译模式下将 CUDA 设备伪装为 ROCM

**getATenDevice()** (lines 135-163)
- 反向转换：DLPack 设备类型转换为 PyTorch Device 对象
- 处理设备 ID 映射

**toScalarType()** (lines 165-263)
- 将 DLDataType 转换回 PyTorch ScalarType
- 根据数据类型代码和位宽确定具体的 Tensor 数据类型

**toDLPack()** (lines 278-306)
- 将 ATen Tensor 包装为 DLManagedTensor
- 规范化步长（stride normalization）处理尺寸为 1 的维度
- 设置数据指针、设备信息、形状和步长

**fromDLPack()** (lines 308-335)
- 从 DLManagedTensor 构建 ATen Tensor
- 两个重载版本：自动管理生命周期 / 自定义删除器
- 处理行优先和带步长的张量

### 关键设计细节

- **ATenDLMTensor 结构**：包装 Tensor 句柄和 DLManagedTensor，用于生命周期管理
- **deleter 回调**：确保 DLPack 张量被释放时正确清理 ATen 资源
- **步长规范化**：避免步长信息不一致的问题

---

### 快速总结

- 双向 Tensor 类型转换（PyTorch ↔ DLPack）
- 支持多种设备后端和数据类型
- 内存生命周期管理和资源清理
- 处理行优先和自定义步长的张量布局
