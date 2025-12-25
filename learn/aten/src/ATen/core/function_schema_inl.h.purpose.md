# function_schema_inl.h 文件分析

这是一个C++内联模板实现文件，主要负责函数schema的参数验证和规范化。

## 核心功能

**checkArg 模板函数** (第8-25行)
- 对单个IValue参数进行类型检查
- 快速路径：如果是Tensor且schema期望Tensor，直接通过
- 特殊处理：空的generic dict被认为有效
- 类型不匹配时抛出TORCH_CHECK错误，包含详细的类型信息

**checkAndNormalizeInputs 模板函数** (第28-76行)
- 验证并规范化函数的所有输入参数
- 检查输入数量是否超过schema定义的参数数量
- 按顺序处理参数：
  - 已提供的positional arguments进行类型检查
  - 缺失的positional arguments从kwargs中查找
  - 仍缺失的参数使用default_value
  - 无法满足的参数抛出错误
- 验证所有kwargs都被使用，未使用的kwargs抛出错误

## 关键特性

- 使用模板设计提供类型灵活性
- 提供详细的错误消息指导用户
- 支持默认参数值和关键字参数
- 执行严格的类型安全检查

---

**主要功能：**
- 对函数调用的参数进行类型验证
- 处理positional和keyword参数的混合
- 应用默认参数值
- 生成详细的参数错误信息
