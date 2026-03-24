# torch 层：Python API 和高级功能

这份文档聚焦在 PyTorch 的最上层——`torch/` 目录。假定你已经理解了 c10 和 ATen 的核心概念（参见 `concepts.md`），这里主要讲 torch 层提供了什么。

## torch 层的职责

torch 层是用户真正接触到的部分，它包含：

1. **Python API**：`import torch` 导入的所有接口
2. **Python-C++ 绑定**：连接 Python 对象和 C++ 对象
3. **Autograd 引擎**：自动微分的核心实现
4. **JIT 编译器**：TorchScript 和图优化
5. **C++ Frontend**：libtorch，纯 C++ 的 PyTorch API
6. **高级功能**：分布式训练、量化、profiling 等

## Python API 层

`torch/__init__.py` 定义了用户可见的所有 API：

```python
import torch

x = torch.tensor([1, 2, 3])
y = x + 1
y.backward()  # autograd
model = torch.jit.script(my_function)  # TorchScript
```

这一层主要是 Python 代码，负责提供 Pythonic 的接口、参数检查和类型转换，然后调用底层 C++ 实现。

## torch/csrc：C++ 核心功能

`torch/csrc` 是 "C source" 的缩写，包含了 PyTorch 的 C++ 核心功能。它不只是简单的绑定层，而是实现了很多核心特性。

### Python-C++ 绑定

把 Python 对象和 C++ 对象连接起来。关键数据结构是 `THPVariable`：

```cpp
// torch/csrc/autograd/python_variable.h
struct THPVariable {
  PyObject_HEAD
  c10::MaybeOwned<at::Tensor> cdata;  // 包装的 C++ tensor
  PyObject* backward_hooks;            // Python 注册的钩子
  PyObject* post_accumulate_grad_hooks;
};
```

当你在 Python 里创建 `torch.Tensor` 时，实际创建的是 `THPVariable` 对象。它内部持有 `at::Tensor`，同时存储 Python 层的元数据（如 backward hooks）。

### Autograd Engine

`torch/csrc/autograd/` 实现自动微分引擎，这是 PyTorch 最核心的功能之一。

当你调用 `loss.backward()` 时，触发这个引擎：构建计算图、拓扑排序、执行反向传播、梯度累积。

重要：autograd 引擎在 **torch 层**，而不是 c10 或 ATen 层。c10 和 ATen 只负责前向计算。

### JIT 编译器和 TorchScript

`torch/csrc/jit/` 实现了 TorchScript 和 JIT 编译器：

- **frontend/**：Python 代码解析和转换
- **ir/**：中间表示（IR）
- **passes/**：编译优化 pass
- **mobile/**：移动端部署支持

`torch.jit.script` 或 `torch.jit.trace` 把 Python 代码转换成独立于 Python 解释器的格式，用于部署和优化。

### C++ Frontend (libtorch)

`torch/csrc/api/` 提供纯 C++ 的 PyTorch API（libtorch），完全不依赖 Python：

```cpp
#include <torch/torch.h>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  auto model = torch::nn::Linear(3, 1);
  auto output = model->forward(tensor);
}
```

C++ API 设计模仿 Python API，但底层直接使用 ATen 和 c10，没有 Python 解释器开销。

## 实例：tensor 加法的完整流程

`c = a + b` 在 torch 层的执行路径：

1. **Python `__add__`** 通过 pybind11 绑定到 C++ 函数
2. 从 `THPVariable` 提取 `at::Tensor`
3. 调用 ATen 的 add 函数（后续流程见 `concepts.md`）
4. Kernel 返回 `at::Tensor`
5. 绑定层创建 `THPVariable` 包装
6. Python 得到 `torch.Tensor` 对象

torch 层负责 Python 绑定，ATen 负责操作实现，c10 负责 dispatch 和内存管理。

## 延伸阅读

- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/) - Edward Yang 的经典文章
- [Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) - dispatcher 机制详解