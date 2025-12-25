## NonSymbolicBC.h 文件分析

这个文件位于PyTorch的ATen库中，主要用于**向后兼容性（Backward Compatibility）**支持。

### 核心目的

该文件声明了一些操作的**非符号化版本（Non-Symbolic）**。PyTorch现代API使用符号整数（Symbolic Ints）来处理张量形状和维度信息，但在某些场景（如静态运行时）中，代码需要直接调用原生操作的非符号版本。

### 主要内容

**导出的公共API函数：**

- `reshape()` - 重新塑形张量
- `narrow()` - 获取张量的连续子集
- `_sparse_coo_tensor_unsafe()` - 创建稀疏COO格式张量
- `nll_loss()` / `nll_loss2d()` - 负对数似然损失计算
- `select()` - 沿指定维度选择元素
- `tensor_split()` - 沿指定维度分割张量

**设计模式：**

- 使用 `at::IntArrayRef` 代替符号化整数参数
- 保持与旧版本的接口兼容性
- 避免重复实现某些向后操作（如 backward 类函数）

---

**总结：**

- 📋 为符号化改造的操作提供非符号版本声明
- 🔄 支持静态运行时等特殊场景的直接调用
- 🛡️ 维持API向后兼容性
- 📌 使用 `IntArrayRef` 处理整数数组参数
