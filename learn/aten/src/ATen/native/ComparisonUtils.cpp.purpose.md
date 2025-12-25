这个文件的主要功能是提供张量元数据的断言/验证功能。

**核心函数：**

- `_assert_match()` - 通用的比较函数，用于验证原始值与期望值是否相等，如果不相等则抛出运行时错误
- `_assert_tensor_metadata_meta_symint()` - 验证张量的符号元数据（symbolic metadata），包括大小、步长、数据类型、设备和布局
- `_assert_tensor_metadata()` - 验证张量的常规元数据，功能同上但使用的是整数大小而非符号大小

**主要用途：**

- 运行时张量属性检查
- 确保张量的大小、步长、dtype、device、layout 符合预期
- 用于调试和验证张量操作的正确性
- 支持可选参数（OptionalArrayRef、optional）允许选择性验证某些属性

**设计特点：**

- 提供两个版本的元数据验证（符号版和常规版）
- 使用模板泛型实现通用的比较逻辑
- 通过 stringstream 提供详细的错误信息
