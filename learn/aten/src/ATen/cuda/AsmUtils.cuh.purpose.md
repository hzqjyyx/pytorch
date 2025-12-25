## AsmUtils.cuh 文件分析

这个文件提供了与 CUDA GPU 汇编指令直接交互的工具函数集合。

### Bitfield 模板类

提供位域操作的两个特化版本：

**unsigned int 版本 (32位)**
- `getBitfield()`: 使用 `bfe.u32` PTX 指令从值中提取指定位置和长度的位
- `setBitfield()`: 使用 `bfi.b32` PTX 指令将数值插入到指定位置
- 主机端退化为标准位运算（左移、掩码操作）

**uint64_t 版本 (64位)**
- `getBitfield()`: 使用 `bfe.u64` PTX 指令
- `setBitfield()`: 使用 `bfi.b64` PTX 指令
- 主机端同样退化为位运算

### Lane 掩码函数（warp 级别操作）

获取当前线程在 warp 中的 lane ID 和掩码：

- `getLaneId()`: 返回当前线程在 32 线程 warp 中的索引 (0-31)
- `getLaneMaskLt()`: 返回比当前 lane ID 小的所有 lane 的掩码
- `getLaneMaskLe()`: 返回比当前 lane ID 小或相等的所有 lane 的掩码
- `getLaneMaskGt()`: 返回比当前 lane ID 大的所有 lane 的掩码
- `getLaneMaskGe()`: 返回比当前 lane ID 大或相等的所有 lane 的掩码

### 关键特点

- **双模式编译**: `#if !defined(__CUDA_ARCH__)` 区分主机端（CPU）和设备端（GPU）代码
- **性能优化**: GPU 端使用原生 PTX 汇编指令；CPU 端使用等价的 C++ 操作
- **ROCm 兼容**: 为 AMD GPU 提供替代实现
- **强制内联**: 所有函数使用 `__forceinline__` 确保无函数调用开销

### 主要用途

- **键值对打包/解包**: 在张量索引或数据结构中存储多个整数值
- **Warp 级归约操作**: 实现线程间通信和数据同步
- **优化的位操作**: 避免条件分支，提升 GPU 计算效率
