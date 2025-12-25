## Distance.cpp/h 主要功能

这两个文件实现了 PyTorch 中的距离计算操作，包括成对距离（pairwise distance）、点对点距离（pdist）、交叉距离（cdist）和余弦相似度。

### 核心函数

**1. `pairwise_distance` (行48-55)**
- 计算两个张量 x1 和 x2 之间的成对距离
- 支持广播机制
- 使用 Lp 范数：`norm(x1 - x2 + eps, p, dim, keepdim)`
- eps 用于数值稳定性

**2. `pdist` (行58-64)**
- 计算单个张量内所有行对之间的距离
- 仅支持 2D 张量
- 要求浮点类型和非负 p 值
- 确保内存连续后调用 `_pdist_forward`

**3. `_pdist_forward` (行244-262)**
- pdist 的实际实现
- 对于 n 行数据，计算 n*(n-1)/2 个距离值
- 通过 `pdist_forward_stub` 分发到具体设备实现

**4. `_euclidean_dist` (行66-79)**
- 欧氏距离计算的第一部分（简化梯度处理）
- 使用矩阵乘法优化：将 ||x1-x2||² 展开为 ||x1||² - 2<x1,x2> + ||x2||²
- 通过拼接技巧用一次矩阵乘法完成：`cat([x1*(-2), ||x1||², 1]) @ cat([x2, 1, ||x2||²]).T`
- 结果 clamp 到非负并开方

**5. `cdist_impl` (行81-148)**
- 计算两个张量集合之间的交叉距离矩阵
- 支持批处理：将批次维度展平为单一维度
- 三种计算模式 (compute_mode)：
  - 0（默认）：p=2 且 r1>25 或 r2>25 时使用矩阵乘法
  - 1：强制使用矩阵乘法（p=2）
  - 2：不使用矩阵乘法
- 特殊情况处理：
  - 空张量返回空结果
  - c1=0（特征维度为0）返回全零
  - p=2 且满足条件时使用 `_euclidean_dist` 优化
  - 其他情况通过 `cdist_stub` 分发

**6. `cdist` (行150-176)**
- cdist 的用户接口
- 验证输入维度（≥2D）和特征维度匹配
- 处理命名张量
- 根据条件选择路径：
  - 空输入或 p≠2：调用 `_cdist_forward`（显式自动求导）
  - p=2 且满足条件：调用 `cdist_impl`（PyTorch 自动推导反向传播）

**7. `cosine_similarity` (行274-330)**
- 计算余弦相似度：`<x1, x2> / (||x1|| * ||x2||)`
- **改进的数值稳定实现**：
  - 旧方案：分别计算分子分母，易溢出且可能 |result| > 1
  - 新方案：先归一化再点积 `<x1/||x1||, x2/||x2||>`
  - 避免显式计算大范数的乘积，保证 |result| ≤ 1
- 使用 `clamp_min(eps)` 避免除零
- 在 NoGradGuard 中修改范数，避免影响梯度

### 设计要点

**批处理策略**（行112-127）
- 将所有批次维度推断并展平为单一批次维度
- 例如：(b1, b2, r, c) → (b1*b2, r, c)
- 使用 `expand` + `contiguous` + `view` 转换

**性能优化**
- p=2 且数据量大时（r>25），用矩阵乘法替代逐元素计算
- 基于性能指标的阈值选择

**分发机制**（Distance.h）
- 通过 `DECLARE_DISPATCH` 声明设备无关接口
- 具体实现在 CPU/CUDA/XPU 后端

**数值稳定性**
- `_euclidean_dist` 中用 `clamp_min(0)` 避免负数开方
- `cosine_similarity` 归一化策略避免溢出

---

**Backward 相关**：
- `_pdist_backward`：pdist 梯度计算，通过 `pdist_backward_stub` 分发
- `_cdist_backward`：cdist 梯度计算，处理广播并通过 `cdist_backward_stub` 分发

**ROCm 相关**：
- 代码中未直接提及 ROCm，但通过 dispatch stub 机制支持（与 CUDA 类似路径）
