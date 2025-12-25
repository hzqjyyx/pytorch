## gemm_kernel_utils.h 核心功能分析

这个文件是 PyTorch 高效注意力机制的 CUDA 实现中的工具头文件，主要提供 GEMM（通用矩阵乘法）内核编译和配置工具。

### 主要组件

**宏定义工具（第 15-82 行）**
- `DISPATCH_TYPES`：根据张量数据类型（float/half/bfloat16）分发到对应的模板实例化分支
- `DISPATCH_BOOL`：编译期条件分发，将运行时布尔值转换为编译期常量
- `DISPATCH_ARCHTAG`：根据 GPU 计算能力（CC >= 50/70/75/80）选择对应的 CUDA 架构标签
- 张量检查宏：`CHECK_NOSPARSE_CONTIGUOUS_CUDA`、`CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA`、`CHECK_ALIGNED_PTR`、`ASSIGN_CHECK_OVERFLOW`

**数学工具函数（第 86-94 行）**
- `ceil_div`：整数向上取整除法
- `align_up`：对齐到指定倍数

**GEMM 类型选择器（第 100-152 行）**
- `DefaultGemmType` 模板：根据 GPU 架构和数据类型自动选择最优 GEMM 配置
  - Volta（SM70）+ FP16：使用 8×8×4 张量核心操作
  - Turing+（SM75+）+ FP16/BF16：使用 16×8×8 张量核心操作
  - Ampere+（SM80+）+ FP32：使用快速 FP32 张量核心
  - 回退方案：SIMT（GPU 核心上的 FMA）

**条件调用工具（第 154-176 行）**
- `call_conditional`：编译期条件模板，在编译时根据布尔常量选择不同函数调用，允许两个分支返回不同类型

**Warp 级工具（第 183-208 行）**
- `warp_uniform`：使用 `__shfl_sync` 从 warp lane 0 广播值，标记为 warp 一致（enables 编译器优化），支持标量和指针类型

### 核心特性
- bullet 点列表：
  - 适配多个 GPU 架构（SM50-SM80+）的高性能 GEMM 配置
  - 支持三种浮点精度（FP32、FP16、BF16）
  - 利用张量核心加速矩阵运算
  - 编译期类型和架构分发，零运行时开销
  - CUDA 内存对齐和连续性检查
  - Warp 级同步原语用于性能优化
