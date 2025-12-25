这个文件是 PyTorch 的库注册系统核心实现，主要负责操作符定义和分发的注册。

**主要功能：**

- **Library 类初始化** - 根据库的类型（DEF/IMPL/FRAGMENT）初始化库，注册到全局 Dispatcher
- **debugString 函数** - 生成调试信息字符串，记录注册代码的文件位置和行号
- **_def 方法** - 定义操作符的 schema，可用于定义或验证操作符
  - 支持两种调用方式：仅 schema 或 schema + CppFunction 实现
  - 检查命名空间一致性，注册 Python 模块（非 mobile 环境）
- **_impl 方法** - 注册操作符的具体实现到特定 dispatch key
  - 解析操作符名称，检查 dispatch key 一致性
  - 支持 REGISTER 或 VERIFY 两种模式
- **_fallback 方法** - 注册全局 fallback 实现，作用于所有运行时 dispatch key
  - 只能在 TORCH_LIBRARY_IMPL 块中调用
  - 跳过 mobile 环境中未使用的 dispatch key
- **_parseNameForLib 方法** - 解析操作符名称，设置默认命名空间
- **_resolve 方法** - 将操作符名称字符串解析为规范的 OperatorName 对象
- **CppFunction 类** - 封装 C++ 函数实现，包含 KernelFunction、C++ 签名和 schema
- **reset 方法** - 清除所有注册的 registrar
