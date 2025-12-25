**ThrustAllocator.h 主要功能分析**

这个文件定义了一个自定义的内存分配器类 `ThrustAllocator`，用于 Thrust 库（NVIDIA 的并行算法库）的 GPU 内存管理。

**核心设计：**

- **目的**：将 Thrust 库内部的 GPU 设备内存分配请求重定向到 PyTorch 的统一 CUDA 内存缓存管理器
- **实现方式**：继承自 Thrust 分配器接口，提供 `allocate()` 和 `deallocate()` 两个方法
- **allocate() 方法**（第14-16行）：
  - 接收请求的内存大小（字节）
  - 调用 `c10::cuda::CUDACachingAllocator::raw_alloc()` 进行实际分配
  - 返回分配的 char* 指针
- **deallocate() 方法**（第18-20行）：
  - 接收指针和大小
  - 调用 `c10::cuda::CUDACachingAllocator::raw_delete()` 释放内存
  - 由 PyTorch 的中央缓存管理器处理释放逻辑

**关键特点：**

- 轻量级包装器，充当 Thrust ↔ PyTorch 内存管理的桥梁
- 确保 Thrust 操作的内存分配纳入 PyTorch 统一的内存管理策略
- 避免 Thrust 绕过 PyTorch 的内存缓存和跟踪系统

**核心职责列表：**

• 为 Thrust 库提供自定义分配器接口
• 将 GPU 内存分配请求转接到 PyTorch CUDA 缓存分配器
• 统一管理 Thrust 和 PyTorch 的 GPU 内存生命周期
• 确保内存分配遵循 PyTorch 的缓存策略和内存跟踪机制
