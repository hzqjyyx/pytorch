# Linear.cpp 主要功能分析

## 核心功能

### 1. linear 函数 (73-122行)
实现 PyTorch 的线性变换 `y = xW^T + b`

**关键优化路径:**
- **2D输入 + bias**: 直接调用融合算子 `addmm(bias, input, weight.t())`
- **3D输入优化**: 
  - 连续内存 → 调用 `_flatten_nd_linear` 展平处理
  - 环境变量 `TORCH_LINEAR_FLATTEN_3D=1` 可强制展平非连续3D输入
- **特殊后端**:
  - MKLDNN tensors → `mkldnn_linear`
  - Mobile端 → XNNPACK加速
- **通用路径**: `matmul(input, weight.t())` + bias加法
  - 对 TensorSubclass 或有前向梯度的tensor使用 `add` (保持复合合规性)
  - 其他情况使用 `add_` 原地操作

### 2. _flatten_nd_linear 辅助函数 (57-70行)
处理高维输入的核心辅助函数:
```
输入: [d1, d2, ..., dn, in_features]
展平: [d1*d2*...*dn, in_features]
计算: addmm(bias, flattened_input, W^T)
恢复: [d1, d2, ..., dn, out_features]
```

### 3. einsum 函数 (255-631行)
实现爱因斯坦求和约定 `torch.einsum("ij,jk->ik", A, B)`

**实现流程:**
1. **解析方程** (288-448行):
   - 支持显式输出 `"ij,jk->ik"` 或隐式输出 `"ij,jk"`
   - 标签映射到 [0,52) 索引 (A-Z, a-z)
   - 处理省略号 `...` 表示任意维度
   
2. **维度对齐** (466-544行):
   - 重复标签取对角线: `diagonal().movedim()`
   - 缺失维度 unsqueeze
   - 按计算顺序 permute

3. **收缩计算** (549-610行):
   - 默认从左到右依次收缩
   - 支持自定义路径参数 `path` (类似opt-einsum)
   - 核心操作: `sumproduct_pair` → 最终归约到批量矩阵乘 `bmm`

4. **输出处理** (612-630行):
   - 多操作数: view去除收缩维 (已为1)
   - 单操作数: sum显式求和

### 4. sumproduct_pair 函数 (145-245行)
einsum的配对归约原语，计算 `(left * right).sum(sum_dims)`

**优化策略:**
- 提前sum掉只出现在单侧的维度
- 维度分类: `lro`(两侧+输出), `lo`(仅左+输出), `ro`(仅右+输出)
- 通过 permute + reshape 转换为 `bmm(left, right)`
- 反向 permute 恢复原始维度顺序

### 5. bilinear 函数 (712-765行)
双线性变换 `y = x1^T W x2 + b`

实现: 
- 展平输入为2D: `[batch, features]`
- 调用 `_trilinear` 核心计算
- reshape回原始批次形状
- 广播加bias

### 6. _trilinear 函数 (637-710行)
三输入爱因斯坦积 `(i1 ⊗ i2 ⊗ i3).sum(sumdim)` 在指定维度展开

**应用:**
- `bilinear` 前向传播
- `bilinear_backward` 梯度计算 (需展开某维度避免内存爆炸)

计算逻辑:
```cpp
for k in unroll_dim:
  buf = sumproduct_pair(i1[k], i2[k], sum_dims_12)
  buf = sumproduct_pair(buf, i3[k], sum_dims_23)
  output[k] += buf
```

### 7. tensordot 函数 (769-825行)
沿指定维度的张量收缩 `tensordot(A, B, dims1=[i,j], dims2=[k,l])`

**实现步骤:**
1. 验证收缩维度尺寸匹配
2. 广播维度(size=1)立即sum
3. 构造permutation: `[非收缩维, 收缩维]`
4. reshape成2D矩阵: `[非收缩积, 收缩积]`
5. 矩阵乘法: `mm(t1, t2)`
6. reshape回目标形状

---

## 其他内容

**ROCm相关:**
- 无 (此文件未包含ROCm特定代码)

**Backward相关:**
- `_trilinear` 设计用于支持 `bilinear_backward` 的梯度计算
- `linear_out` 提供输出预分配版本,反向传播可能使用
