- **量化方案枚举定义**：`QScheme` 是一个 `uint8_t` 枚举，定义了 5 种量化类型
  - `PER_TENSOR_AFFINE`（0）：按张量仿射量化
  - `PER_CHANNEL_AFFINE`（1）：按通道仿射量化
  - `PER_TENSOR_SYMMETRIC`（2）：按张量对称量化
  - `PER_CHANNEL_SYMMETRIC`（3）：按通道对称量化
  - `PER_CHANNEL_AFFINE_FLOAT_QPARAMS`（4）：按通道仿射量化（浮点参数）

- **常量别名**：为各枚举值提供 `constexpr` 常量别名（如 `kPerTensorAffine`），便于代码引用

- **编译时枚举数量**：`COMPILE_TIME_NUM_QSCHEMES = 5`，记录支持的量化方案总数

- **字符串转换函数**：`toString()` 函数将 `QScheme` 枚举值转换为对应的字符串表示（例如 `PER_TENSOR_AFFINE` → `"per_tensor_affine"`）

- **Python 同步**：注释说明需与 `torch/nn/_qscheme.py` 保持一致
