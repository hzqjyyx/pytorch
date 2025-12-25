## Formatting.h - 头文件声明

定义了用于输出和格式化的公开接口：

1. **c10 命名空间中的操作符和函数**：
   - `operator<<(std::ostream&, Backend)` - 后端类型的流输出
   - `operator<<(std::ostream&, const Scalar&)` - 标量值的流输出
   - `toString(const Scalar&)` - 标量转字符串

2. **at 命名空间中的操作符和函数**：
   - `operator<<(std::ostream&, const DeprecatedTypeProperties&)` - 类型属性的流输出
   - `print(std::ostream&, const Tensor&, int64_t linesize)` - Tensor详细打印
   - `operator<<(std::ostream&, const Tensor&)` - Tensor流输出的快捷方式（调用print，行宽80）
   - `print(const Tensor&, int64_t linesize)` - Tensor打印到标准输出

## Formatting.cpp - 实现文件

### 核心功能模块：

**Scalar 输出处理** (lines 15-41)：
- 根据标量类型（浮点、复数、布尔、SymInt、SymFloat、整数）分别处理并输出
- 提供字符串转换接口

**格式化控制工具** (lines 46-67)：
- `defaultfloat()` - C++标准浮点格式
- `FormatGuard` - RAII模式保存/恢复流格式状态

**Tensor 打印核心** (lines 73-342)：

1. `__printFormat()` - 计算最优显示格式
   - 判断是否为整数模式
   - 计算指数范围
   - 根据数值大小选择科学计数法或定点记数法
   - 返回缩放因子和列宽

2. `__printIndent()` - 缩进输出

3. `__printMatrix()` - 2D矩阵打印
   - 处理超宽矩阵的分列显示
   - 应用缩放和精度格式化
   - 按指定行宽排列列

4. `__printTensor()` - 3D及以上多维张量打印
   - 遍历高维索引
   - 逐个矩阵切片输出

5. `print()` - 主打印函数
   - 处理特殊情况：未定义张量、稀疏张量、量化张量、MKLDNN、MPS张量
   - 转换为CPU上的float64进行打印
   - 根据维度调用相应的打印函数
   - 输出量化参数和自动求导信息

---

### 功能总结：

- **Scalar 输出** - 多类型标量的流式输出和转字符串
- **格式管理** - 流状态保存/恢复，防止格式污染
- **智能格式选择** - 根据数值范围自动选择科学计数法或定点法
- **多维张量显示** - 支持0D到ND的张量，自动处理超宽矩阵的分列显示
- **特殊张量支持** - 量化张量、稀疏张量、不同设备张量的转换和打印
- **求导信息输出** - 显示前向梯度（tangent）信息
