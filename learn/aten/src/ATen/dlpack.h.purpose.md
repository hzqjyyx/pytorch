这个文件定义了 DLPack 协议的 C 接口，用于在不同深度学习框架之间零拷贝共享张量数据。

## 核心数据结构

**DLDevice (lines 101-109)**
- 描述张量所在的设备
- `device_type`: 设备类型枚举（CPU、CUDA、Metal、Vulkan 等）
- `device_id`: 设备索引，对于 CPU/pinned memory 设为 0

**DLDataType (lines 149-162)**
- 描述张量的数据类型
- `code`: 类型码（整数、浮点、复数、布尔等）
- `bits`: 位宽（8/16/32/64）
- `lanes`: 向量化通道数，用于 SIMD 类型

**DLTensor (lines 167-210)**
- 核心张量结构，不负责内存管理
- `data`: 数据指针（理论上应 256 字节对齐，但实际很多库未遵守）
- `device`: 设备信息
- `ndim`: 维度数
- `dtype`: 数据类型
- `shape`: 形状数组
- `strides`: 步长数组（可为 NULL 表示紧凑行主序）
- `byte_offset`: 数据起始的字节偏移

**DLManagedTensor (lines 219-232)**
- 带生命周期管理的张量包装
- `dl_tensor`: 实际的 DLTensor
- `manager_ctx`: 原始框架的上下文指针
- `deleter`: 析构函数指针，用于通知原框架释放资源

## 设计要点

1. **零拷贝共享**: 通过指针直接共享内存，避免数据复制
2. **框架无关**: 纯 C 接口，任何语言/框架都能使用
3. **生命周期管理**: 通过 deleter 回调实现跨框架的引用计数
4. **设备抽象**: 统一表示 CPU、GPU、专用加速器等 17 种设备类型

## 版本信息
- DLPack 版本: 80 (line 19)
- ABI 版本: 1 (line 22)

---

**其他设备类型**: OpenCL, Vulkan, Metal, VPI, OneAPI, WebGPU, Hexagon, MAIA

**Backward 兼容性**: 通过 ABI 版本号和 C 接口保证
