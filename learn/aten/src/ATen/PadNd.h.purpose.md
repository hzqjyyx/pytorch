这个文件定义了 PyTorch ATen 库中的填充模式枚举。

**主要功能：**

- 定义 `padding_mode` 枚举类，包含4种填充模式
- `reflect`: 反射填充（镜像边界值）
- `replicate`: 复制填充（重复边界值）
- `circular`: 循环填充（环绕边界值）
- `constant`: 常数填充（用指定常数填充）
- 位于 `at` 命名空间内，供 ATen 张量操作使用
