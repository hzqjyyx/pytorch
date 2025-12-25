## RangeFactories 模块分析

这个模块实现了PyTorch中生成数值序列张量的工厂函数。

### 核心功能

**linspace** - 生成等间距序列
- 4个重载版本处理Tensor/Scalar的start和end参数
- 验证输入维度为0
- 支持1个和多个步长的特殊情况优化
- 使用`linspace_stub`分发到具体实现（CPU/CUDA等）

**logspace** - 生成对数等间距序列
- 4个重载版本，类似linspace
- base参数控制对数底数（默认10）
- 为复数和实数类型提供分开的实现
- 使用对称计算方式（从中间点向两端计算）避免累积误差

**range/arange** - 生成算术序列
- `range_out`：标准range实现，步长必须非零
- `range_out_no_step`：简化版本，默认步长为1
- `arange_out`：包含错误检测和警告，如果输出张量大小不匹配会调整

### 关键实现细节

- **contiguity处理**：非连续张量先转为连续，计算后再复制回去
- **Meta设备支持**：检测kMeta设备直接返回，不进行实际计算
- **并行化**：使用`at::parallel_for`和`GRAIN_SIZE`进行多线程计算
- **数据类型分发**：用`AT_DISPATCH_*`宏处理不同标量类型
- **精度处理**：logspace对复数和浮点使用累加类型，range使用double中间值

---

### 要点总结

- linspace：等间距序列（支持Tensor/Scalar参数）
- logspace：对数等间距序列（支持复数类型）
- range：算术序列（固定步长）
- arange：算术序列（包含验证警告）
- 使用dispatch stubs分发到设备特定实现
- 支持并行化计算和非连续张量处理
- Meta设备短路优化
