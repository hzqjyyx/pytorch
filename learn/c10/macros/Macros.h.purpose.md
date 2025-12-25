这个文件是 C10 库的核心宏定义集合，主要提供以下功能：

## 核心功能

**1. 编译器和平台检测**
- 检测和配置不同编译器（Clang、GCC、MSVC）的特性
- 通过 `__has_attribute`、`__has_feature` 等检测编译器能力
- 定义平台相关宏（Android、iOS、移动平台等）

**2. CUDA/HIP 设备代码支持**
- `C10_HOST_DEVICE`：标记可在 CPU/GPU 执行的函数
- `C10_DEVICE`、`C10_HOST`：分别标记设备端和主机端函数
- `C10_LAUNCH_BOUNDS`：CUDA kernel 启动参数配置
- `CUDA_MAX_THREADS_PER_BLOCK`、`CUDA_MAX_THREADS_PER_SM`：根据架构调整线程限制

**3. UndefinedBehaviorSanitizer (UBSAN) 控制**
- 提供一系列 `__ubsan_ignore_*` 宏来禁用特定未定义行为检测
- 包括浮点除零、整数溢出、指针溢出、类型转换溢出等

**4. AddressSanitizer (ASAN) 检测**
- `C10_ASAN_ENABLED`：检测是否启用 ASAN
- 支持 Clang 和 GCC 的 ASAN 检测方式

**5. 内联和可见性控制**
- `C10_ALWAYS_INLINE`：强制内联函数
- `C10_NOINLINE`：禁止内联
- `C10_ERASE`：结合强制内联和隐藏可见性，用于优化
- `C10_ATTR_VISIBILITY_HIDDEN`：隐藏符号可见性

**6. 分支预测优化**
- `C10_LIKELY(expr)`：提示表达式大概率为真
- `C10_UNLIKELY(expr)`：提示表达式大概率为假
- 使用 `__builtin_expect` 优化分支预测

**7. 实用工具宏**
- `C10_CONCATENATE`：拼接宏参数
- `C10_STRINGIZE`：将宏参数转为字符串
- `C10_ANONYMOUS_VARIABLE`：生成唯一标识符（使用 `__COUNTER__` 或 `__LINE__`）
- `C10_DISABLE_COPY_AND_ASSIGN`：禁用拷贝构造和赋值运算符

**8. C++ 属性宏**
- `C10_NODISCARD`：`[[nodiscard]]` 标记返回值不应被忽略
- `C10_UNUSED`：`[[maybe_unused]]` 抑制未使用警告
- `C10_USED`：标记符号必须保留

**9. CUDA Kernel 断言**
- `CUDA_KERNEL_ASSERT`：即使在 Release 模式（NDEBUG）也执行的断言
- `CUDA_KERNEL_ASSERT_MSG`：带消息的断言
- 跨平台支持（Windows 使用 `_wassert`，Unix 使用 `__assert_fail`）

**10. 命名空间定义**
```cpp
namespace c10 {}
namespace c10::cuda {}
namespace c10::hip {}
namespace c10::xpu {}
```
并将 c10 命名空间别名到 caffe2 和 at，保证向后兼容

**11. 编译器诊断控制**
- `C10_CLANG_DIAGNOSTIC_PUSH/POP`：Clang 警告控制
- `C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED`：跨编译器忽略特定警告

**12. 移动平台优化**
- 检测 Android/iOS 平台
- `C10_ALWAYS_INLINE_UNLESS_MOBILE`：移动平台不强制内联以减小二进制体积

---

**ROCm 相关**（概述）：
- `C10_HIP_HOST_DEVICE`、`C10_WARP_SIZE` 适配 AMD GPU
- ROCm 默认禁用 kernel assert

**向后兼容**（概述）：
- `CONSTEXPR_EXCEPT_WIN_CUDA` 等宏处理旧版 Windows/CUDA 限制
- caffe2/at 命名空间别名
