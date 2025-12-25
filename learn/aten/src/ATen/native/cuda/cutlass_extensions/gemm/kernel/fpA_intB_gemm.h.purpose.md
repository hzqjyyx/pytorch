这个文件实现了一个支持**浮点数A矩阵×整数B矩阵**的GEMM（通用矩阵乘法）CUDA kernel模板。

## 核心功能

### 1. 模板结构 `GemmFpAIntB`
主模板类接受5个参数：
- `Mma_`: 线程块级别的矩阵乘累加运算
- `Epilogue_`: 收尾阶段处理
- `ThreadblockSwizzle_`: 线程块调度策略
- `KernelArch`: 目标架构（Sm70/75/80）
- `SplitKSerial`: 是否支持split-K串行归约

### 2. 量化支持
- **ElementA**: 浮点数类型（FP16/FP32）
- **ElementB**: 整数类型（INT4/INT8）
- **ElementScale**: 缩放因子类型，用于反量化B矩阵
- 引入 `ref_scale` 专门存储量化缩放参数

### 3. 数据布局适配
支持B矩阵的特殊布局：
- RowMajor (kInterleave=1)
- ColumnMajor interleaved (kInterleave≥1)
- 通过 `kInterleave` 参数控制交错因子

### 4. Arguments & Params
**Arguments** (用户接口):
```cpp
- problem_size: GEMM维度(M,N,K)
- ref_A, ref_B, ref_scale: 输入张量引用
- ref_C, ref_D: 输出张量引用
- serial_split_k_factor: split-K分割数
- gather/scatter_indices: 支持gather/scatter操作
```

**Params** (kernel参数):
包含迭代器参数、网格配置、信号量等运行时信息。

### 5. Kernel执行流程 (`KernelRunner`)

**主循环阶段** (lines 317-389):
```
1. 计算线程块tile偏移 (threadblock_swizzle)
2. 构造A/B/Scale迭代器，设置起始位置
3. 执行MMA运算: mma(iterations, accumulators, iter_A, iter_B, iter_scale)
   - 在计算过程中动态应用scale进行反量化
```

**Epilogue阶段** (lines 391-452):
```
1. 构造输出迭代器C/D
2. 处理split-K同步（使用Semaphore）
3. 执行epilogue融合操作（如bias、activation）
4. 写回结果到D
```

### 6. 条件编译优化
通过 `__CUDA_ARCH__` 宏和模板特化避免无关架构代码编译：
- Sm70: Volta (CUDA Arch 7.0)
- Sm75: Turing (CUDA Arch 7.5)
- Sm80: Ampere (CUDA Arch 8.0+)

### 7. Split-K支持
当 `kSplitKSerial=true` 时：
- 将K维度分割成多个子问题
- 使用semaphore进行线程块间同步
- 串行累加partial results到最终结果

### 8. 对齐检查
`can_implement()` 验证所有tensor引用满足内存对齐要求（32/64 bytes或AccessType对齐）。

---

**其他特性（简要）**：
- 支持gather/scatter索引访问（稀疏场景）
- 通过workspace管理临时内存（当前实现返回0）
- 使用shared memory union减少内存占用（main_loop和epilogue复用）
- 兼容CUTLASS GemmUniversal接口（batch_stride_D等参数）
