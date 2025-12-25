# jiterator.cu 和 jiterator.h 主要功能

## 核心功能

这两个文件实现了 PyTorch 的 **Jiterator (JIT Iterator)** 系统，用于在运行时动态编译和启动 CUDA 内核，特别针对 pointwise 操作。

## jiterator.h - 接口定义

提供单一公共接口：

```cpp
c10::SmallVector<at::Tensor> CompileAndLaunchKernel(
  const std::string& code_string,      // 用户定义的 CUDA 代码
  const std::string& kernel_name,      // 内核名称
  const int num_outputs,               // 输出张量数量
  const c10::SmallVector<at::Tensor>& tensors,  // 输入张量
  const c10::SmallVector<at::Scalar>& extra_args, // 额外标量参数
  bool return_by_ref);                 // 是否通过引用返回
```

当 `AT_USE_JITERATOR()` 未定义时，提供一个抛出错误的占位实现。

## jiterator.cu - 核心实现

### 1. CompileAndLaunchKernel (330-364行)

**职责**：入口函数，设置 TensorIterator 并调用内核

**流程**：
- 创建 TensorIterator 配置（内存重叠检查、类型提升、安全转换等）
- 添加输出和输入张量
- 调用 `jitted_gpu_kernel_dynamic` 执行实际的 JIT 编译和启动
- 返回输出张量

### 2. jitted_gpu_kernel_dynamic (286-324行)

**职责**：动态 GPU 内核的入口点，处理设备检查和 32 位索引

**关键逻辑**：
- 验证所有张量在 CUDA 设备上
- 处理空张量情况
- 对于超过 32 位索引的张量，递归调用子迭代器
- 判断是否需要动态类型转换（当输入/输出类型与 common_dtype 不同时）
- 调用 `jitted_gpu_kernel_dynamic_impl`

### 3. jitted_gpu_kernel_dynamic_impl (193-278行)

**职责**：根据张量特性选择 4 种内核启动策略

**四种情况**：

| Case | Dynamic Casting | Contiguous | 使用的函数 |
|------|----------------|-----------|----------|
| 1 | ✗ | ✓ | `launch_jitted_vectorized_kernel_dynamic` |
| 2 | ✗ | ✗ | `launch_jitted_unrolled_kernel_dynamic` |
| 3 | ✓ | ✓ | `launch_jitted_unrolled_kernel_dynamic` |
| 4 | ✓ | ✗ | `launch_jitted_unrolled_kernel_dynamic` |

**准备工作**：
- 创建 ArrayVariant 存储数据指针
- 对于非连续张量，创建 OffsetCalculator
- 对于需要动态转换的，创建 LoadWithCastVariant 和 StoreWithCastVariant

### 4. launch_jitted_vectorized_kernel_dynamic (14-121行)

**职责**：启动向量化内核（用于 Case 1 - 连续且无需类型转换）

**优化策略**：
- 计算线程工作大小 (tws) 和向量化大小 (vec_size = 1/2/4)
- 向量化可以一次处理多个元素，提升内存带宽利用率

**缓存机制**：
- 使用 stringstream 生成缓存键（包含输入数、输出数、函数、类型、vec_size、设备索引）
- 静态 unordered_map 缓存已编译的内核
- 双重检查锁定 (double-checked locking) 避免重复编译

**代码生成**：
- 调用 `at::cuda::jit::generate_code` 生成完整的 CUDA 代码
- 向量化内核名称添加后缀：`name + "_vectorized" + vec_size`
- 调用 `jit_pwise_function` 使用 NVRTC 编译

**参数打包**：
- 向量化版本：3 个基础参数 (N, data_ptr, scalar_val) + extra_args
- 非向量化版本：7 个基础参数 (N, data_ptr, input_offset_calc, output_offset_calc, loader, storer, scalar_val) + extra_args

### 5. launch_jitted_unrolled_kernel_dynamic (123-191行)

**职责**：启动展开循环内核（用于 Case 2/3/4）

**特点**：
- 不使用向量化
- 需要 offset calculators 处理非连续内存
- 需要 loader/storer 函数指针处理类型转换
- 缓存键包含 contiguous 和 dynamic_casting 标志

**参数固定为 7 个**：
```cpp
args[0] = &N;              // 元素总数
args[1] = data_ptr;        // 数据指针数组
args[2] = ic_ptr;          // 输入偏移计算器
args[3] = oc_ptr;          // 输出偏移计算器
args[4] = l_ptr;           // 加载函数指针
args[5] = s_ptr;           // 存储函数指针
args[6] = &scalar_val;     // 标量值（未使用但保留）
// + extra_args...
```

### 6. 关键数据结构和辅助类

- **ArrayVariant**: 存储多个张量的数据指针数组
- **OffsetCalculatorVariant**: 计算非连续张量的内存偏移
- **TrivialOffsetCalculatorVariant**: 连续张量的简化版本
- **LoadWithCastVariant / StoreWithCastVariant**: 处理不同数据类型的加载/存储

## 工作流程总结

```
用户代码 (code_string)
    ↓
CompileAndLaunchKernel
    ↓
创建 TensorIterator (类型提升、内存检查等)
    ↓
jitted_gpu_kernel_dynamic (设备验证、32位索引检查)
    ↓
jitted_gpu_kernel_dynamic_impl (判断 4 种情况)
    ↓
┌─────────────────┬──────────────────┐
│   连续 + 无转换  │    其他情况       │
│        ↓        │        ↓          │
│ vectorized 内核 │  unrolled 内核    │
└─────────────────┴──────────────────┘
    ↓
生成缓存键并查找缓存
    ↓
缓存未命中 → generate_code → NVRTC 编译
    ↓
打包参数并启动 CUDA 内核
    ↓
返回输出张量
```

## 设计优势

- **动态性**: 运行时接受任意用户定义的 CUDA 代码
- **性能优化**: 连续内存使用向量化，最大化带宽
- **类型灵活**: 自动处理类型转换
- **缓存**: 避免重复编译相同内核
- **可扩展**: 支持可变数量的输入/输出和额外参数

---

**ROCm 相关**: 无（文件中未涉及）

**Backward 相关**: 无（纯前向计算框架）
