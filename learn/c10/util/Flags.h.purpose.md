## C10_UTIL_FLAGS_H 文件分析

这是一个**命令行标志（flags）的跨平台支持库**，为 C10 提供便携式的参数解析机制。

### 核心设计

文件通过条件编译分为两种实现：
1. **基于 gflags** (`C10_USE_GFLAGS` 定义时) - 使用谷歌的 gflags 库
2. **轻量级自定义实现** - 当 gflags 不可用时的备选方案

### 主要 API

**命名空间 c10 中的公开函数：**
- `SetUsageMessage()` - 设置 `--help` 的帮助文本
- `UsageMessage()` - 获取已设置的帮助文本  
- `ParseCommandLineFlags()` - 解析命令行参数，移除已处理的参数
- `CommandLineFlagsHasBeenParsed()` - 检查标志是否已解析

### 定义和声明宏

支持五种基本类型，每种都有 `DEFINE` 和 `DECLARE` 两种宏：
- `C10_DEFINE_int / C10_DECLARE_int`
- `C10_DEFINE_int64 / C10_DECLARE_int64`
- `C10_DEFINE_double / C10_DECLARE_double`
- `C10_DEFINE_bool / C10_DECLARE_bool`
- `C10_DEFINE_string / C10_DECLARE_string`

访问方式统一为 `FLAGS_<flag_name>`

### 自定义实现细节（非 gflags）

- 定义了 `C10FlagParser` 基类，包含模板化的 `Parse()` 方法
- 使用注册表模式 (`C10FlagsRegistry`) 动态注册各个标志解析器
- 通过宏生成匿名命名空间中的解析器类，在全局范围内自动注册

### 关键特性

- **Python 兼容性**：由于 Python 无法修改 main，推荐设置合理的默认值
- **可见性管理**：当 C10 以共享库编译时，重新定义 gflags 的 EXPORT 宏以确保全局变量正确导出
- **版本兼容**：处理 gflags 2.0 前后的命名空间变化（google vs gflags）

---

**核心职责：**
- ✓ 提供统一的跨平台命令行参数接口
- ✓ 支持 gflags 和轻量级两种后端
- ✓ 通过宏简化标志定义和声明
- ✓ 在 Python 环境中安全运行
