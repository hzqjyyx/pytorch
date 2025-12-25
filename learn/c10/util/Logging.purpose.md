## PyTorch C10 Logging 模块功能分析

这两个文件实现了 PyTorch 核心库 c10 的日志和错误处理系统，提供了统一的日志接口和断言机制。

### 核心功能模块

**1. 日志系统初始化和配置**

- 支持两种后端：Google Glog (C10_USE_GLOG) 或自定义实现
- 通过环境变量 `TORCH_CPP_LOG_LEVEL` 设置日志级别（INFO/WARNING/ERROR/FATAL）
- `initLogging()` 和 `InitCaffeLogging()` 进行初始化
- `FLAGS_caffe2_log_level` 控制最小日志级别

**2. 断言和错误抛出机制**

`CAFFE_ENFORCE` 系列宏提供运行时断言：
- `CAFFE_ENFORCE(condition, msg)` - 条件不满足时抛出异常
- `CAFFE_ENFORCE_EQ/NE/LT/LE/GT/GE` - 二元比较断言，失败时显示两边的值
- `CAFFE_ENFORCE_FINITE` - 检查数值有限性
- `CAFFE_ENFORCE_WITH_CALLER` - 包含 this 指针的断言

实现细节：
- `ThrowEnforceNotMet()` 构造 `c10::Error` 异常并抛出
- 如果设置 `FLAGS_caffe2_use_fatal_for_enforce=true`，使用 LOG(FATAL) 直接终止而非抛异常
- 自动捕获调用栈 backtrace（通过 `GetFetchStackTrace()`）

**3. 调用栈追踪**

- `SetStackTraceFetcher()` 允许自定义 backtrace 获取函数
- 默认使用 `get_lazy_backtrace(frames_to_skip=1)` 延迟获取
- `PyTorchStyleBacktrace` 类格式化异常信息，包含源位置和调用栈

**4. API 使用追踪**

用于收集 API 调用统计：
- `C10_LOG_API_USAGE_ONCE(event)` 宏确保每个事件只记录一次（利用静态变量）
- `SetAPIUsageLogger()` 设置自定义 logger
- `LogAPIUsage(event)` 记录事件
- `LogAPIUsageMetadata()` 记录带元数据的事件
- 支持通过 `PYTORCH_API_USAGE_STDERR` 环境变量启用调试输出

**5. 事件采样处理**

- `EventSampledHandler` 接口允许注册事件处理器
- `InitEventSampledHandlers()` 批量注册
- `C10_LOG_EVENT_SAMPLED(event, model_id, args)` 宏触发采样日志
- 使用全局 registry 管理处理器，线程安全（mutex 保护）

**6. 自定义日志实现（非 Glog 模式）**

当未使用 Google Glog 时（c10/util/Logging.cpp:369-500）：
- `MessageLogger` 类实现日志消息构造
- 格式：`[rank][级别MMDD HH:MM:SS.ns 文件:行] 消息`
- 输出到 stderr，严重级别 > INFO 自动 flush
- Android 平台使用 `__android_log_print`
- FATAL 级别调用 `DealWithFatal()` 终止程序

**7. 分布式训练支持**

- `SetGlobalRank(rank)` 设置全局 rank
- 日志消息自动包含 `[rankN]` 前缀（rank != -1 时）
- `DDPLoggingData` 结构体存储分布式训练元数据
- `SetPyTorchDDPUsageLogger()` 和 `LogPyTorchDDPUsage()` 专门处理 DDP 日志

### ROCm/Backward 相关
- 无 ROCm 特定代码
- 无反向传播相关内容
