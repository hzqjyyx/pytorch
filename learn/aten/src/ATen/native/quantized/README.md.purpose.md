## 量化算子实现指南

这个文档是PyTorch中实现低级量化内核（quantized kernel）的入门指南。

### 核心内容

**量化文件夹的作用：**
- 存储低级量化内核的实现
- 内核注册在 `torch::_ops` 命名空间
- 操作量化的 `at::Tensor` 数据类型

**实现新量化算子的步骤：**

1. **实现量化函数** — 编写具体的C++函数，使用 `AT_DISPATCH_QINT_TYPES` 宏处理所有量化类型
   - `SCALAR_TYPE` — 量化张量的标量类型（如 `kQInt8`）
   - `scalar_t` — 量化数据类型（如 `qint8`）
   - `underlying_t` — 底层POD数据类型（如 `int8_t`）

2. **定义schema** — 在 `library.cpp` 中用 `TORCH_LIBRARY` 宏声明算子签名

3. **注册实现** — 使用 `TORCH_LIBRARY_IMPL` 宏将实现绑定到QuantizedCPU后端

4. **[可选] 添加到native_functions.yaml** — 如果签名与非量化版本相同，可注册以支持更多集成

5. **编译配置** — 更新 `TARGETS` 和 `CMakeLists.txt` 以包含新文件

**使用量化算子：**

- **Python** — 通过 `torch._ops.ops.quantized.xand()` 调用，建议放在 `torch/ao/nn/quantized/functional.py`
- **C++** — 通过 Dispatcher 查找和调用（非官方支持）

### 要点总结

- 量化内核几乎总是位于 `ATen/native/quantized/cpu` 文件夹
- 使用 `TensorIterator` 和 `cpu_kernel` 处理张量迭代和计算
- `AT_DISPATCH_QINT_TYPES` 宏确保代码适配所有量化类型
- 新算子需要三个注册步骤：函数实现、schema定义、后端注册
- 编译系统会自动处理 `quantized/cpu` 下的文件
