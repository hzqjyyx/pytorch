这个文件实现了 CUDA 设备上的原子操作，为不同数据类型提供线程安全的读-修改-写操作。

## 核心设计

**AtomicFPOp 模板结构**
- 为 Half、BFloat16、double 提供通用的原子浮点运算框架
- 使用 Compare-And-Swap (CAS) 循环实现：读取旧值 → 计算新值 → 尝试交换 → 失败则重试
- Half/BFloat16 需要特殊处理：通过对齐到 32 位边界访问，使用位操作提取/插入 16 位值

**整数类型原子操作宏**
- `ATOMIC_INTEGER_IMPL(NAME)` 生成 1/2/4/8 字节整数的原子操作实现
- 小于 4 字节的类型：通过 32 位对齐地址 + 位掩码操作实现
- 4/8 字节类型：直接使用 atomicCAS

## 主要实现的操作

**gpuAtomicAdd 系列**
- 整数类型 (int8/16/32/64, uint8, bool)：大部分通过自定义 CAS 实现，int32 直接用 CUDA 内置 atomicAdd
- 半精度浮点 (Half/BFloat16)：旧架构用自定义实现，新架构 (CC≥7.0/8.0) 用硬件原生支持
- 单/双精度浮点：使用 CUDA 内置 atomicAdd（旧架构的 double 需要自定义实现）
- 复数类型：分别对实部和虚部调用原子加

**gpuAtomicMul/Max/Min**
- 通过 `ATOMIC_INTEGER_IMPL` 宏为各种整数类型生成实现
- 浮点类型使用 AtomicFPOp 框架 + 相应的 lambda 函数
- Max/Min 使用 `safe_max/safe_min` 处理 NaN（NaN 传播优先）

**gpuAtomicAddNoReturn**
- 无返回值版本的原子加，AMD MI100 (gfx908) 上的 float 可使用优化指令 `atomicAddNoRet`

## 兼容性处理

- 提供 `atomicAdd` 重载（直接调用 `gpuAtomicAdd`），供外部扩展（如 torchvision）直接调用
- 旧 CUDA 架构 (<6.0) 需要自定义 double 的 atomicAdd
- 使用编译时条件判断选择硬件原生实现或软件模拟

---

**ROCm 相关**：HIP 平台的特定实现（如 `__atomic_fetch_add`、`unsafeAtomicAdd`、NaN 处理差异）

**Backward 兼容**：为不同 CUDA/HIP 编译器版本和硬件架构提供条件编译分支
