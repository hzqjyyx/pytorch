## DefaultDtype 功能分析

这两个文件实现了 PyTorch 中的**默认数据类型管理系统**。

### DefaultDtype.h（头文件）
定义了四个公开 API 函数：
- `set_default_dtype()` - 设置全局默认数据类型
- `get_default_dtype()` - 获取当前默认数据类型（TypeMeta 格式）
- `get_default_dtype_as_scalartype()` - 获取默认数据类型（ScalarType 格式）
- `get_default_complex_dtype()` - 获取对应的默认复数数据类型

### DefaultDtype.cpp（实现文件）
维护三个静态全局变量：
1. `default_dtype` - 默认数据类型（初始值：float）
2. `default_dtype_as_scalartype` - 默认数据类型的标量类型版本
3. `default_complex_dtype` - 对应的复数数据类型

`set_default_dtype()` 函数核心逻辑：
- 更新 `default_dtype` 和 `default_dtype_as_scalartype`
- 根据设置的数据类型自动映射对应的复数类型：
  - Half → ComplexHalf
  - Double → ComplexDouble
  - 其他 → ComplexFloat（默认）

### 核心功能点
- **全局状态管理** - 通过静态变量存储应用级别的默认数据类型配置
- **类型同步** - 设置标量类型时自动更新复数类型，保持一致性
- **API 隔离** - 通过函数接口暴露，隐藏内部实现细节

### 主要用途
- 允许用户改变 PyTorch 张量创建时的默认精度（如从 float32 改为 float64）
- 确保创建复数张量时使用与标量类型匹配的复数精度
