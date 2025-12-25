这个文件定义了一个模板元程序结构，用于为不同的迭代器类型生成"残差优先"（Residual Last）版本。

**主要功能：**

- **模板特化机制**：通过模板特化将基础迭代器类型映射到对应的残差优先版本
- **PredicatedTileIterator 转换**：将 `PredicatedTileIterator` 转换为 `PredicatedTileIteratorResidualLast`
- **PredicatedTileAccessIterator 转换**：将 `PredicatedTileAccessIterator` 转换为 `PredicatedTileAccessIteratorResidualLast`
- **内存高效注意力优化**：支持 CUTLASS 框架中的内存高效多头注意力（Mem-Eff Attention）计算
- **访问模式变换**：通过"残差优先"策略优化张量访问顺序和缓存局部性

**技术特征：**

- 使用 CUTLASS 库的迭代器抽象
- 支持不同的数据布局和访问模式
- 支持 Gather 操作的可选启用
- 模板参数包括形状、元素类型、线程映射等底层硬件参数
