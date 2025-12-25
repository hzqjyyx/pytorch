# Unfold3d 核心功能

这两个文件实现了3D卷积操作中的 **im2col/col2im** 转换，将3D输入张量展开成适合矩阵乘法的列格式（或反向操作）。

## 主要操作

### 1. Unfold3dCopy (im2col)
**作用**: 将3D输入张量按滑动窗口展开成列矩阵

**过程**:
- 输入: `(C, X_D, X_H, X_W)` 形状的3D数据
- 输出: `(C×kernel_d×kernel_h×kernel_w, Y_D×Y_H×Y_W)` 形状的2D矩阵
- 对每个通道的每个卷积核位置，提取对应的空间窗口数据

**实现路径**:
```
Unfold3dCopyCPU (aten/src/ATen/native/Unfold3d.cpp:435)
  └─> Unfold3dCopyKernelImpl (line:224)
       ├─> pad=0: Unfold3dZeroPaddingCopyKernelImpl (line:180)
       └─> pad>0: 逐元素处理边界填充 (line:266-299)
```

### 2. Unfold3dAcc (col2im)
**作用**: 将展开的列矩阵累加回3D张量格式（梯度反向传播用）

**过程**:
- 输入: `(C×kernel_d×kernel_h×kernel_w, Y_D×Y_H×Y_W)` 展开矩阵
- 输出: `(C, X_D, X_H, X_W)` 累加后的3D数据
- 将重叠区域的值累加到对应位置（卷积梯度需要）

**实现路径**:
```
Unfold3dAccCPU (aten/src/ATen/native/Unfold3d.cpp:483)
  └─> Unfold3dAccKernelImpl (line:356)
       ├─> pad=0: Unfold3dZeroPaddingAccKernelImpl (line:303)
       └─> pad>0: 逐元素累加有效区域 (line:397-430)
```

## 核心优化策略

### 1. 零填充特化
无填充时调用优化版本 (`ZeroPadding` 变体):
- 使用 `MatCopy`/`MatAdd` 批量处理连续内存
- `stride_w=1` 时用 `memcpy` 直接复制行
- `stride_w>1` 时逐元素处理跨步访问

### 2. MKL 加速
当 `AT_MKL_ENABLED()` 时，`float`/`double` 特化使用:
- `mkl_somatcopy`/`mkl_domatcopy`: 矩阵拷贝
- `mkl_somatcopy2`/`mkl_domatcopy2`: 带跨步的矩阵拷贝
- `mkl_somatadd`/`mkl_domatadd`: 矩阵加法
- `cblas_saxpy`/`cblas_daxpy`: 向量加法（跨步场景）

### 3. 并行化
- `at::parallel_for` 在外层循环并行处理不同通道/卷积核位置
- Copy 操作: 按 `C×kernel_d×kernel_h×kernel_w` 并行
- Acc 操作: 按通道 `C` 并行（需要先清零目标区域）

## 关键参数映射

```
X_D/X_H/X_W: 输入深度/高度/宽度
Y_D/Y_H/Y_W: 输出深度/高度/宽度
kernel_d/h/w: 卷积核尺寸
stride_d/h/w: 滑动步长
pad_d/h/w: 填充大小
```

## 辅助函数

**IsAGeZeroAndALtB** (line:17): 边界检查优化
- 用 `unsigned` 比较同时检查 `a >= 0 && a < b`

**MatCopy** (line:22, 29): 矩阵拷贝
- 基础版: 连续行拷贝
- 跨步版: 支持自定义行/列跨步

**MatAdd** (line:48, 59): 矩阵累加 `Y += X`
- 对应两种跨步模式

---

**不相关内容（忽略）**:
- ROCm/HIP 相关: 文件中未涉及
- Backward 相关: `Unfold3dAcc` 本身就是为反向传播设计，但文件中只实现前向操作的基础工具
