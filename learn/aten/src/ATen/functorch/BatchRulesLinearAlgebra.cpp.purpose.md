这个文件实现了 PyTorch functorch 库中线性代数操作的批处理规则（batch rules）。当使用 `vmap` 对线性代数函数进行向量化映射时，这些规则定义了如何处理带有批次维度的张量。

## 核心功能

**批次维度管理**
- 将批次维度移动到张量前面（`moveBatchDimToFront`）
- 确保张量具有批次维度（`ensure_has_bdim`）
- 处理不同输入张量批次维度的对齐和广播

**矩阵乘法类操作的批处理规则**
- `dot_batch_rule`: 向量点积，处理 1D 向量的批处理
- `mv_batch_rule`: 矩阵-向量乘法，处理 2D×1D 的批处理
- `mm_batch_rule`: 矩阵-矩阵乘法，处理 2D×2D 的批处理
- `bmm_batch_rule`: 批量矩阵乘法，处理 3D×3D 的批处理
- `tv_batch_rule`: 张量-向量乘法的通用规则

关键设计原则（Note [Batching rules for matmul-like operators]）：避免不必要的维度扩展，如果只有一个参数是 BatchedTensor，不要在另一个参数上扩展批次维度。

**分解策略**
某些操作通过分解为更简单的操作来实现批处理：
- `addmv_decomp`: `beta * input + alpha * (mat @ vec)`
- `addmm_decomp`: `beta * self + alpha * (mat1 @ mat2)`
- `addbmm_decomp`: `beta * input + alpha * bmm(batch1, batch2)`
- `baddbmm_decomp`: 类似 addbmm 但保留批次维度
- `vdot_decomp`: 共轭点积

**线性代数求解器的批处理**
- `linalg_lu_solve_batch_rule`: LU 分解求解
- `linalg_lu_unpack_batch_rule`: LU 分解解包
- `linalg_lu_factor_ex_batch_rule`: LU 因子分解
- `solve_ex_batch_rule`: 通用线性系统求解，包含复杂的秩对齐逻辑
- `cholesky_solve_batch_rule`: Cholesky 分解求解
- `linalg_lstsq_batch_rule`: 最小二乘求解

**模板化批处理规则生成器**
- `LinalgCheckMatrixUnaryRuleHelper`: 为单输入矩阵操作生成批处理规则
  - 检查输入至少是 2D（矩阵）
  - 支持 1-4 个输出的操作
  - 自动移动批次维度并调用底层函数

- `LinalgCheckMatrixBinaryRuleHelper`: 为双输入矩阵操作生成批处理规则
  - 检查两个输入都至少是 2D
  - 使用 `_binary_pointwise_helper` 对齐批次维度
  - 支持 1-2 个输出的操作

**宏系统**
使用宏批量注册批处理规则：
- `LINALG_CHECK_MATRIX_UNARY_ONE_OUT`: 单输入单输出（如 cholesky）
- `LINALG_CHECK_MATRIX_UNARY_TWO_OUT`: 单输入双输出（如 linalg_qr）
- `LINALG_CHECK_MATRIX_UNARY_THREE_OUT`: 单输入三输出（如 linalg_svd）
- `LINALG_CHECK_MATRIX_BINARY_ONE_OUT`: 双输入单输出（如 linalg_solve_triangular）

**特殊操作**
- `householder_product_batch_rule`: Householder 变换
- `matrix_exp_batch_rule`: 矩阵指数
- `cross_batch_rule`: 向量叉积
- `pinv_batch_rule`: Moore-Penrose 伪逆
- `_linalg_check_errors_batch_rule`: 错误检查

**注意力机制批处理**
- `_scaled_dot_product_flash_attention_batch_rule`: Flash Attention
- `_scaled_dot_product_efficient_attention_batch_rule`: Efficient Attention
- `_scaled_dot_product_cudnn_attention_batch_rule`: cuDNN Attention

这些规则处理 dropout 的随机性检查，将批次维度展平到第一维，然后重新整形输出。

## 已注册的操作

通过 `TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m)` 注册的操作包括：
- cholesky, cholesky_inverse, linalg_cholesky_ex
- linalg_eig, linalg_inv_ex, linalg_qr, linalg_slogdet
- linalg_ldl_factor_ex, geqrf, triangular_solve
- _linalg_det, _linalg_eigh, _linalg_svd
- bmm, dot, mv, mm
- 各种求解器和分解操作

---

**ROCm 相关**: 无

**Backward 相关**: 
- `solve_ex_batch_rule` 中有关于 autograd 保存张量的注释（NOTE [ solve_ex Batch Rule Contiguity ]）
- 需要确保批处理前后的连续性检查一致，以便 autograd（特别是 JVP）能正确工作
