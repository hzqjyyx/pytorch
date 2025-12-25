# c10/util/Exception 的主要功能分析

这两个文件实现了 PyTorch 核心库 c10 的异常处理和警告系统。

## 核心类和功能

### 1. Error 类 - 主要异常类型
**c10/util/Exception.h:30-115**

- 继承自 `std::exception`，是 ATen 的主要错误类
- 包含三个核心数据：
  - `msg_`: 实际错误消息
  - `context_`: 错误上下文堆栈（从具体到抽象）
  - `backtrace_`: C++ 调用栈
  - `caller_`: 调试用的指针（可用于识别触发异常的算子）

- 支持三种构造方式：
  - PyTorch 风格：`Error(SourceLocation, msg)` (c10/util/Exception.cpp:63)
  - Caffe2 风格：`Error(file, line, condition, msg)` (c10/util/Exception.cpp:21-38)
  - 基础构造：`Error(msg, backtrace, caller)` (c10/util/Exception.cpp:11-14)

- 关键方法：
  - `what()`: 返回完整错误消息（含 backtrace）(c10/util/Exception.cpp:65-76)
  - `what_without_backtrace()`: 返回不含 backtrace 的消息
  - `add_context()`: 向上下文堆栈添加新消息 (c10/util/Exception.cpp:88-97)
  - `compute_what()`: 格式化错误消息，单上下文时折叠到一行，多上下文时每行一个 (c10/util/Exception.cpp:40-59)

- 使用 `OptimisticLazy<std::string>` 延迟计算 `what_`，避免提前计算 backtrace

### 2. 专用异常类型
**c10/util/Exception.h:238-320**

通过继承 Error 创建的特定用途异常，转换到 Python 时会映射到对应类型：

- `IndexError` → Python IndexError
- `ValueError` → Python ValueError  
- `TypeError` → Python TypeError
- `NotImplementedError` → Python NotImplementedError
- `LinAlgError` → Python LinAlgError
- `OutOfMemoryError`
- `SyntaxError`
- `DistError` (分布式训练基础错误)
  - `DistBackendError`
  - `DistStoreError`
  - `DistNetworkError`
- `ErrorAlwaysShowCppStacktrace`: 总是显示 C++ 堆栈的错误

### 3. Warning 类 - 警告系统
**c10/util/Exception.h:117-160, c10/util/Exception.cpp:199-244**

- 支持两种警告类型：
  - `UserWarning`
  - `DeprecationWarning`
  
- 包含信息：
  - `type_`: 警告类型
  - `source_location_`: 触发位置
  - `msg_`: 警告消息
  - `verbatim_`: 是否原样输出（不添加 Python 上下文）

- `verbatim` 标志的设计考虑：C++ 警告可能与 Python 用户代码脱节，非 verbatim 时允许警告处理器添加 Python 上下文

### 4. WarningHandler - 警告处理器
**c10/util/Exception.h:169-174, c10/util/Exception.cpp:139-196**

- 默认实现：打印到 stderr (c10/util/Exception.cpp:246-251)
- 使用线程本地存储管理当前处理器 (c10/util/Exception.cpp:147-166)
  - `ThreadWarningHandler::warning_handler_` 是 thread_local
  - 未设置时回退到 `base_warning_handler_`（全局静态单例）
  
- 提供的工具：
  - `set_warning_handler()` / `get_warning_handler()`: 设置/获取处理器
  - `WarningHandlerGuard`: RAII 风格临时替换处理器
  - `set_warnAlways()` / `get_warnAlways()`: 控制 `TORCH_WARN_ONCE` 行为
  - `WarnAlways`: RAII 风格临时设置 warn_always

### 5. 宏定义 - 用户 API
**c10/util/Exception.h:337-692**

核心检查宏：

- `TORCH_CHECK(cond, ...)`: 用户输入检查，失败抛出 Error (c10/util/Exception.h:580-588)
  - 实现：调用 `torchCheckFail()` (c10/util/Exception.cpp:101-115)
  
- `TORCH_INTERNAL_ASSERT(cond, ...)`: 内部不变量断言 (c10/util/Exception.h:418-428)
  - 实现：调用 `torchInternalAssertFail()` (c10/util/Exception.cpp:117-135)
  - Debug only 版本：`TORCH_INTERNAL_ASSERT_DEBUG_ONLY` (仅 debug 模式检查)

- `TORCH_RETHROW(e, ...)`: 添加上下文后重新抛出异常 (c10/util/Exception.h:377-381)

类型特定检查宏（抛出特定异常类型）：
- `TORCH_CHECK_LINALG` → LinAlgError
- `TORCH_CHECK_INDEX` → IndexError
- `TORCH_CHECK_VALUE` → ValueError
- `TORCH_CHECK_TYPE` → TypeError
- `TORCH_CHECK_NOT_IMPLEMENTED` → NotImplementedError
- `TORCH_CHECK_IF_NOT_ON_CUDA`: 仅在 host 代码检查，CUDA/HIP 代码中为空

警告宏：
- `TORCH_WARN(...)`: 发出 UserWarning (c10/util/Exception.h:663)
- `TORCH_WARN_DEPRECATION(...)`: 发出 DeprecationWarning (c10/util/Exception.h:665-666)
- `TORCH_WARN_ONCE(...)`: 仅警告一次（静态变量实现）(c10/util/Exception.h:681-686)

辅助宏：
- `C10_THROW_ERROR(err_type, msg)`: 直接抛出指定类型异常
- `C10_BUILD_ERROR(err_type, msg)`: 构造异常对象（不抛出）
- `TORCH_CHECK_ARG(cond, argN, ...)`: 参数检查专用

### 6. 实现细节

**消息格式化** (c10/util/Exception.cpp:40-59):
```cpp
compute_what(bool include_backtrace) {
  oss << msg_;
  if (context_.size() == 1) {
    oss << " (" << context_[0] << ")";  // 单上下文：折叠
  } else {
    for (c : context_) { oss << "\n  " << c; }  // 多上下文：多行
  }
  if (include_backtrace) { oss << "\n" << backtrace_; }
}
```

**延迟计算优化** (c10/util/Exception.cpp:78-86):
- `what_` 使用 `OptimisticLazy` 包装，首次访问时才计算
- `refresh_what()` 重置 `what_`，但立即计算 `what_without_backtrace_`
- 避免在不需要 backtrace 时计算（性能优化）

**线程安全性**:
- Warning handler 使用 thread_local 存储，每线程独立
- `add_context()` 有 O(n²) 性能问题注释，但标注为非并发调用的公开方法

**条件编译**:
- `STRIP_ERROR_MESSAGES`: 移除详细错误消息（减小二进制大小）
- `STANDALONE_TORCH_HEADER`: `TORCH_CHECK` 抛出 `std::runtime_error` 而非 `c10::Error`
- `DISABLE_WARN`: 禁用所有警告

**工具函数**:
- `GetExceptionString()`: 返回异常类型+消息，支持 RTTI (c10/util/Exception.cpp:253-259)
- `torchCheckMsgImpl()`: 重载处理不同参数类型的消息构造

## Backward/ROCm 相关内容
- 无
