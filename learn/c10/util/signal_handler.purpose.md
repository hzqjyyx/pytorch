# Signal Handler 信号处理器

这两个文件实现了 PyTorch 的信号处理机制，主要用于优雅地响应进程信号和在致命错误时生成堆栈跟踪。

## 核心功能

### 1. SignalHandler - 基础信号处理

处理 `SIGINT` (Ctrl+C) 和 `SIGHUP` (终端挂断) 两种信号：

- **引用计数式管理**: `hookedUpCount` 确保多个 SignalHandler 实例可以共存，只在第一个实例时安装处理器，最后一个销毁时卸载
- **原子计数器**: `sigintCount` 和 `sighupCount` 记录信号触发次数
- **链式调用**: 保存之前的信号处理器 (`previousSigint/previousSighup`)，触发时先执行自己的逻辑再调用原处理器
- **每实例检测**: 每个 SignalHandler 对象维护自己的计数快照 (`my_sigint_count_/my_sighup_count_`)，通过 `GotSIGINT()/GotSIGHUP()` 检测自创建后是否收到过信号

使用场景：训练循环中定期调用 `CheckForSignals()` 判断是否需要提前终止。

### 2. FatalSignalHandler - 致命信号处理（仅 Linux）

处理会导致程序崩溃的信号：`SIGABRT`, `SIGINT`, `SIGILL`, `SIGFPE`, `SIGBUS`, `SIGSEGV`。

**核心机制 - 多线程堆栈跟踪**:

1. 主线程收到致命信号时，设置全局标志 `fatalSignalReceived = true`
2. 遍历 `/proc/self/task` 找到进程的所有线程
3. 向其他线程发送 `SIGUSR2` 信号，让它们打印各自的堆栈
4. 使用条件变量等待每个线程完成堆栈输出（超时 2 秒）
5. 当前线程直接调用 `stacktraceSignalHandler(false)` 打印自己的堆栈
6. 恢复原信号处理器并重新抛出信号

**关键实现细节**:

- **leaky singleton**: `getInstance()` 返回指针而不释放，避免模块析构顺序问题
- **独立栈空间**: 使用 `SA_ONSTACK` 标志，防止栈溢出导致的二次崩溃
- **SIGUSR2 复用**: 拦截 `SIGUSR2` 用于线程间通信，非致命信号时仍调用原处理器
- **双重检查**: `fatalSignalReceived` 标志确保 `SIGUSR2` 处理器知道是在响应崩溃还是正常信号

### 3. 条件编译

- **Apple (macOS)**: 仅支持基础 `SignalHandler`
- **Linux**: 支持完整的 `FatalSignalHandler`
- **其他平台**: SignalHandler 方法都返回空实现

## 典型输出格式

```
SIGSEGV(11), PID: 12345, Thread 12346: 
 [backtrace content from c10::get_backtrace()]
```

## 与 ROCm/Backward 相关内容

- signal_handler.cpp:224-226: 代码注释提到在 ROCm 环境中某些线程可能收不到 `SIGUSR2`，因此使用 `wait_until` 而非 `wait` 并设置超时
