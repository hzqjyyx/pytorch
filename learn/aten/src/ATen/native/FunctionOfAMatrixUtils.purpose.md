这两个文件实现了一个线性组合计算的通用工具函数。

**核心功能：**

`_compute_linear_combination` 函数用于计算张量的加权线性组合：
- 输入：`coefficients` [m, n] 矩阵 + `input` [n, ...] 张量
- 输出：[m, ...] 张量，其中 `output[i, ...] = Σ(coefficients[i, j] * input[j, ...])`
- 相当于进行矩阵-张量乘法，但保持高维张量的形状

**实现特点：**

1. **两个版本**：
   - `_compute_linear_combination()`：创建新输出张量
   - `_compute_linear_combination_out()`：写入现有张量

2. **张量重排策略** (FunctionOfAMatrixUtils.cpp:51-93)：
   - 通过 `unsqueeze()` 和 `as_strided()` 将三个张量重排为统一维度 (input.dim() + 1)
   - 将第二维设置为 stride=0（广播），避免同步和原子操作
   - 保证确定性结果（autograd 要求）

3. **分发机制**：
   - 使用 `DECLARE_DISPATCH` / `DEFINE_DISPATCH` 宏
   - `_compute_linear_combination_stub` 作为函数指针，由不同后端（CPU/CUDA/等）实现

4. **TensorIterator 集成**：
   - 配置迭代器遍历输出和输入张量
   - 禁用内存重叠检查（输出张量已调整为 0-stride）
   - 传递步长信息到后端内核处理第二维求和

**关键设计考量：**

- 避免原子操作，确保 GPU 并行化效率
- 支持复数类型（scalar_t 和 complex 的关联）
- 内存布局灵活性（Contiguous 内存格式）

**总结：**
- 矩阵-张量乘法的高效实现
- 张量重排确保确定性和并行化
- 后端无关设计（通过分发机制支持多设备）
