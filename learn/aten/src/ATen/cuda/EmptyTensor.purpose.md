## 文件功能分析

这两个文件实现了CUDA张量的空张量创建功能。

**EmptyTensor.h** 定义了三组函数的声明：
- `empty_cuda()` 函数族：创建空CUDA张量
- `empty_strided_cuda()` 函数族：创建带指定步幅的空CUDA张量

每组都有三个重载版本，参数逐步递增：
1. 最详细版本：接收具体的size、dtype、device、memory_format
2. 中间版本：接收optional参数（dtype、layout、device、pin_memory、memory_format）
3. 简化版本：接收TensorOptions对象（包含所有配置信息）

**EmptyTensor.cpp** 实现了这些函数的逻辑：

`empty_cuda()` 的工作流程（EmptyTensor.cpp:8-20）：
- 延迟初始化CUDA设备
- 获取设备信息并验证是CUDA设备
- 设置设备Guard确保操作在正确设备上执行
- 获取CUDA设备分配器
- 调用 `at::detail::empty_generic()` 创建张量

`empty_strided_cuda()` 的工作流程（EmptyTensor.cpp:48-60）：
- 逻辑与 `empty_cuda()` 类似
- 额外接收stride参数
- 调用 `at::detail::empty_strided_generic()` 创建指定步幅的张量

各重载版本的作用（EmptyTensor.cpp:23-46, 63-92）：
- optional参数版本进行参数验证后调用具体版本
- TensorOptions版本解析选项后调用optional参数版本

### 核心要点

- **目的**：为CUDA设备提供统一的空张量创建接口
- **设计**：三层参数重载，逐步简化API使用
- **关键操作**：设备管理、分配器获取、DispatchKey设置
- **验证**：pin_memory只允许CPU张量、layout检查（Strided/Jagged）
- **委托**：最终调用 `empty_generic()`/`empty_strided_generic()` 完成实际创建
