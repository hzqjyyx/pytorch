## VmapModeRegistrations.cpp 主要功能

这个文件为 PyTorch 的 `vmap` (vectorized map) 功能注册 `DispatchKey::VmapMode` 的内核实现，主要目的是**禁止在 vmap 内部调用随机操作**。

### 核心设计理念

**随机操作的歧义性问题** (lines 13-21)：
```cpp
vmap(lambda t: torch.rand(1))(torch.zeros(5))
```
这个操作存在语义不明确：
- 应该返回 5 个相同的随机数？
- 还是返回 5 个不同的随机数？

由于尚未确定语义，PyTorch 临时禁止在 vmap 内使用随机操作。

### 实现机制

**1. 错误处理函数** (lines 23-31)：
- `unsupportedRandomOp<>()` - 处理返回新 Tensor 的随机操作
- `unsupportedRandomOp_<>()` - 处理 in-place 随机操作（带 `_` 后缀）
- 两者都抛出统一错误信息，建议在 vmap 外执行随机操作

**2. Fallback 注册** (lines 33-35)：
```cpp
TORCH_LIBRARY_IMPL(_, VmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
```
为所有命名空间注册 fallthrough，让非随机操作正常通过到下一个 dispatch key。

**3. 随机操作拦截** (lines 37-111)：
在 `aten` 命名空间下为所有随机操作注册错误处理函数，覆盖：

- **分布采样**：`bernoulli`, `cauchy_`, `exponential_`, `geometric_`, `log_normal_`, `multinomial`, `normal`, `poisson`
- **随机初始化**：`random_`, `rand`, `randn`, `randint`, `randperm`
- **类似操作**：`rand_like`, `randn_like`, `randint_like`
- **通用操作**：`uniform_`

每个操作的所有重载版本（包括 out 变体、generator 变体等）都被显式注册。

### 关键注释

**DispatchKey::VmapMode vs DispatchKey::Batched** (lines 8-11)：
- `VmapMode`：vmap 内所有 Tensor 都会触发此 key，用于禁用随机操作
- `Batched`：实际的批处理规则注册在这个 key 下

---

**其他相关内容**：
- 无 ROCm 相关内容
- 无 Backward/梯度计算相关内容
