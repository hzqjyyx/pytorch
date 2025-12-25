## 文件主要功能

**one_hot()** 函数将索引张量转换为独热编码表示。

### 核心逻辑：

1. **类型检查** (line 18)
   - 输入张量必须是 `kLong` 类型

2. **Meta/Fake Tensor 优化路径** (line 21-29)
   - 对于 Meta Tensor 或 Python dispatch，使用函数式实现
   - 自动推断类数：`num_classes = max(input) + 1`
   - 通过 `arange` + `eq` + `to(kLong)` 生成独热编码

3. **空张量处理** (line 35-42)
   - 空张量需要显式指定 `num_classes`，否则报错
   - 返回正确形状的空张量

4. **非空张量处理** (line 44-66)
   - 验证类值非负 (line 48)
   - 验证类值小于 `num_classes` (line 56)
   - 若 `num_classes == -1`，自动推断为 `max(input) + 1`
   - 创建零张量，通过 `scatter_` 在对应位置设置 1

### 关键特点：

- **形状变换**：输入 shape `(...)` → 输出 shape `(..., num_classes)`
- **设备适配**：CUDA/MPS/XLA/PrivateUse1 使用设备端断言避免同步
- **两条实现路径**：Meta Tensor 用函数式，普通张量用 scatter 实现

### 用途总结：

- **编码**：将类别索引转换为独热编码向量
- **应用场景**：分类任务中的标签编码、embedding 预处理等
