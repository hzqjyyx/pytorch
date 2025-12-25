## TensorTransformations.cpp 主要功能

**flip 函数** (36-108行)
- 沿指定维度翻转张量
- 通过修改数据指针和步幅来实现高效翻转，避免复制数据
- 使用 TensorIterator 处理多维情况
- 对量化张量 (QUInt4x2, QUInt2x4) 进行检查限制

**roll 函数** (110-132行)
- 沿指定维度循环移位张量元素
- 优化单维度情况：通过 narrow 和 cat 操作实现
- 处理 Python 和 C++ 取模运算的符号差异

**rot90 函数** (134-178行)
- 在指定两个维度平面内旋转 90 度倍数
- 支持 k 参数控制旋转次数（模4处理）
- 内部通过组合 flip 和 transpose 操作实现

**atleast_Nd 函数族** (192-256行)
- atleast_1d：确保张量至少 1 维（0维 reshape 为 [1]）
- atleast_2d：确保张量至少 2 维（0维 [1,1]，1维 unsqueeze）
- atleast_3d：确保张量至少 3 维（0维 [1,1,1]，1维/2维相应 unsqueeze）
- 每个函数有单张量和多张量重载版本

**其他**
- **fliplr/flipud**：便捷函数，分别翻转第1维和第0维
- **chalf**：转换张量数据类型为 ComplexHalf

## TensorTransformations.h 主要功能

**roll_common 函数** (13-33行)
- 通用 roll 实现，处理多维度情况
- 空 dims 时将张量展平后 roll 再恢复形状
- 递归处理多维 roll：先处理第一维，再递归处理剩余维度

---

- **张量几何变换**：flip、roll、rot90 实现
- **维度确保**：atleast_1d/2d/3d 保证最小维度
- **性能优化**：指针+步幅操作避免数据复制
- **多维支持**：通过 TensorIterator 和递归处理高维张量
- **量化感知**：对特定量化格式的检查
