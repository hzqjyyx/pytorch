## Backend.h 核心功能

这是一个**遗留的(legacy)** 枚举类定义文件，用于表示 PyTorch 旧式代码生成系统中支持的后端类型。一个 Backend 大致对应 (设备类型 × 布局) 的笛卡尔积组合，但仅包含实际有内核实现的组合。

### 关键概念

**Backend 枚举** (c10/core/Backend.h:29-66)
- 定义了约 30 种后端类型：CPU、CUDA、XPU、IPU、MAIA、XLA、Metal、Vulkan 等
- 包含密集、稀疏 (Sparse)、CSR 稀疏、量化 (Quantized) 等不同布局变体
- 不包含 dtype 信息，仅关注设备和布局

**设计缺陷与替代方案**
- 不支持开放注册 (open registration)：添加新后端需要修改此枚举，无法树外扩展
- 已被 `DispatchKey` 替代，后者支持开放注册

### 主要函数

**1. dispatchKeyToBackend()** (c10/core/Backend.h:68-143)
- 将 `DispatchKey` 转换为对应的 `Backend`
- 处理普通 key 和 autograd key 的映射（如 `DispatchKey::CPU` 和 `DispatchKey::AutogradCPU` 都映射到 `Backend::CPU`）

**2. backendToDispatchKey()** (c10/core/Backend.h:145-218)
- 反向转换：Backend → DispatchKey
- 使用 switch-case，未知 backend 抛异常

**3. backendToDeviceType()** (c10/core/Backend.h:220-282)
- Backend → DeviceType 转换
- 多个 backend 可能映射到同一设备类型（如 CPU、SparseCPU、QuantizedCPU 都映射到 DeviceType::CPU）

**4. toString()** (c10/core/Backend.h:284-357)
- 返回 backend 的字符串表示
- 未知类型返回 "UNKNOWN_BACKEND"

**5. 布局判断函数**
- `isSparse()` (c10/core/Backend.h:359-371)：判断是否为 COO 稀疏格式
- `isSparseCsr()` (c10/core/Backend.h:373-385)：判断是否为 CSR 稀疏格式

### ROCm 相关

- `Backend::HIP`、`SparseHIP`、`SparseCsrHIP` 对应 AMD ROCm 平台

### Backward/Autograd 相关

- `dispatchKeyToBackend()` 处理 autograd 相关的 DispatchKey（如 `AutogradCPU`、`AutogradCUDA` 等），统一映射到对应的基础 Backend
