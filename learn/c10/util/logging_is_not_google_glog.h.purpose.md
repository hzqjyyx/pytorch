这个文件是 PyTorch 的 C10 库中的日志模块，当未使用 Google Glog 时作为备选实现。主要功能包括：

**核心组件：**

- **日志级别常量**：定义了 GLOG_FATAL、GLOG_ERROR、GLOG_WARNING、GLOG_INFO 四个日志级别
- **MessageLogger 类**：负责实际的日志输出，接收文件名、行号和日志级别，内部使用 stringstream 缓冲日志内容
- **LoggerVoidify 类**：用于条件日志宏中丢弃日志流，避免编译器警告

**日志宏定义：**

- LOG(n) / VLOG(n)：基础日志宏，支持日志级别过滤
- LOG_IF(n, condition) / VLOG_IF(n, condition)：条件日志
- LOG_AT_FILE_LINE()：支持覆盖源位置信息
- DLOG(n) / DCHECK()：仅在调试模式下生效的日志和检查

**检查宏：**

- CHECK()：布尔条件检查，失败时输出错误信息并终止
- TORCH_CHECK_OP()：值比较检查（EQ、NE、LE、LT、GE、GT）
- TORCH_DCHECK_*：调试模式下的检查宏
- TORCH_CHECK_NOTNULL()：指针非空检查

**STL 容器支持：**

- 为 std::vector、std::map、std::set、std::pair 实现流操作符重载
- PrintSequence()：序列输出函数，最多显示 100 个元素

**关键特性：**

- 支持编译时日志级别阈值过滤（CAFFE2_LOG_THRESHOLD）
- 致命日志自动调用 abort() 终止程序
- 调试模式下的宏在优化编译中生成零开销代码
