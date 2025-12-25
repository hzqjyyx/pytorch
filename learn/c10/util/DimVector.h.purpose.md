# DimVector.h 文件分析

这个头文件定义了两个核心类型别名，用于在PyTorch中表示张量的维度信息：

## 关键定义

**DimVector** - 基于 `SmallVector<int64_t, kDimVectorStaticSize>` 的容器
- 用于存储张量的尺寸或步长信息
- 使用固定大小的栈内存优化（避免频繁堆分配）

**SymDimVector** - 基于 `SmallVector<c10::SymInt, kDimVectorStaticSize>` 的容器
- 存储符号整数（SymInt）形式的维度信息
- 支持动态形状推导（如导出模型时的符号维度）

**kDimVectorStaticSize** 常量
- 值来自 `C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE`
- 定义了栈内能存储的最大元素数量

## 主要功能

• 为张量维度/步长信息提供高效的容器抽象

• 通过 SmallVector 优化常见情况下的内存分配（栈存储）

• 支持两种维度表示：具体值（int64_t）和符号值（SymInt）

• 与 c10 核心库的其他数据结构（SizesAndStrides）紧密集成

• 简化张量元数据的类型定义，提高代码可读性
