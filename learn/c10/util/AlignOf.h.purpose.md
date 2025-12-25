## 文件功能概述

`AlignOf.h` 是一个跨平台的类型对齐计算工具库，来自 LLVM，被 PyTorch 的 c10 库修改使用。

## 核心组件

**AlignedCharArray**
- 模板结构体，用于创建具有指定对齐要求的字符缓冲区
- 非 MSVC 环境：直接使用 `alignas()` 关键字
- MSVC 环境：需要特殊处理，预定义常见对齐值（1、2、4、8、16、32、64、128 字节）的特化版本

**detail::AlignerImpl 和 detail::SizerImpl**
- AlignerImpl：通过组合多个类型成员，计算这些类型组合后的对齐需求
- SizerImpl：联合体，包含多个类型的数组，用于计算组合大小

**AlignedCharArrayUnion**
- 继承自 AlignedCharArray
- 通过 `alignof()` 获取 AlignerImpl 的对齐要求
- 通过 `sizeof()` 获取 SizerImpl 的大小
- 支持最多 10 个类型，可通过 placement new 在其缓冲区中构造任意类型

## 主要用途

- **内存对齐**：确保数据按 CPU/编译器要求对齐，提高访问效率
- **类型安全存储**：为不同类型提供统一的对齐存储空间
- **跨平台兼容**：处理 MSVC 和其他编译器的对齐差异

## 关键要点

- 被修改点：将 LLVM 的 `LLVM_ALIGNAS` 替换为标准 `alignas`
- 目标场景：需要动态构造多种类型且要求特定对齐的场合
- 设计限制：当前支持最多 10 个类型组合
