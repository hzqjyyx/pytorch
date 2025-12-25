这个文件定义了 XPU（英特尔 GPU）设备属性的数据结构。通过使用 X-macro 模式，将设备属性分组定义，最终生成统一的 `DeviceProp` 结构体。

**主要组成部分：**

- **AT_FORALL_XPU_DEVICE_PROPERTIES** - 标准 SYCL 设备属性（名称、类型、厂商、版本、可用性、内存配置、计算能力等）

- **AT_FORALL_XPU_EXT_DEVICE_PROPERTIES** - 英特尔 GPU 扩展属性（EU 数量、EU 子片数、SIMD 宽度、硬件线程等）

- **AT_FORALL_XPU_DEVICE_ASPECT** - 设备能力标志（fp16、fp64、64位原子操作支持）

- **AT_FORALL_XPU_EXP_CL_ASPECT** - 实验性 SYCL 功能（bfloat16 转换、矩阵乘加运算、2D 块 IO 支持）

- **AT_FORALL_XPU_EXP_DEVICE_PROPERTIES** - 实验性设备属性（设备架构，仅在编译器版本 ≥ 2025.0000 时启用）

**核心作用：**

- 通过宏展开自动生成 `DeviceProp` 结构体的所有成员变量
- 提供统一的接口查询 SYCL 设备的硬件特性和能力
- 支持版本条件编译，适配不同 SYCL 编译器版本
