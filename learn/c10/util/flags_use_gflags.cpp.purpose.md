这个文件实现了 PyTorch 中 C10 库的命令行标志解析功能，基于 gflags 库。主要内容：

- **条件编译**：仅在定义了 `C10_USE_GFLAGS` 宏时编译此文件
- **SetUsageMessage()**：设置命令行帮助信息，如果已设置过则忽略重复设置
- **UsageMessage()**：获取当前的使用说明文本，通过调用 `gflags::ProgramUsage()` 实现
- **ParseCommandLineFlags()**：解析命令行参数，处理空参数情况，委托给 gflags 库执行实际解析
- **CommandLineFlagsHasBeenParsed()**：查询标志是否已解析，目前简单返回 true（gflags 不提供直接查询方式）
- **导出方式**：所有函数使用 `C10_EXPORT` 宏导出为公共 API

本质上是对 gflags 库的 C10 命名空间包装层。
