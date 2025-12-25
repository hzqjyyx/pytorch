这是一个 PyTorch ATen 库的核心操作注册机制的测试文件,用于验证算子(operator)注册系统的各种功能。

## 主要测试内容

### 1. 基础算子注册
测试通过 `TORCH_LIBRARY` 宏注册自定义算子的能力:
- 定义算子签名(schema),如 `my_op(Tensor self, int arg) -> Tensor`
- 注册算子实现函数
- 验证调用和调度机制

### 2. 多种实现注册方式
测试算子可以通过不同方式注册实现:
- `.impl()` - 直接注册函数实现
- `.impl_UNBOXED()` - 注册非装箱实现
- Lambda 函数
- 函数指针
- 类方法

### 3. Dispatch Key 系统
测试针对不同后端/场景的算子分发:
- CPU 实现注册 (`DispatchKey::CPU`)
- CUDA 实现注册 (`DispatchKey::CUDA`)
- AutogradCPU 等自动微分相关 key
- Fallback 机制(未注册特定实现时的回退行为)

### 4. 算子重载(Overload)
测试同一算子的多个变体:
- 默认重载
- 命名重载,如 `my_op.overload_name`
- Schema 解析和匹配

### 5. 内核(Kernel)的生命周期管理
测试注册和注销机制:
- `registerKernel()` 和 `deregisterKernel()`
- 多次注册同一算子的错误处理
- 注册句柄(RegistrationHandleRAII)的 RAII 管理

### 6. Schema 验证
测试算子签名定义的正确性:
- 参数类型匹配
- 返回值类型验证
- 可变参数处理
- Alias 分析(别名注解 `a!`, `b` 等)

### 7. Catchall Kernel
测试通配符内核注册:
- 注册可以处理所有 dispatch key 的实现
- 与特定 key 实现的优先级关系

### 8. 函数式 API 注册
除了 `TORCH_LIBRARY` 宏,还测试纯函数式 API:
- `c10::RegisterOperators().op()`
- 链式调用注册多个算子

### 9. 错误处理
测试各种非法操作的检测:
- 重复注册
- Schema 不匹配
- 未实现算子的调用
- 类型错误

### 10. Boxing/Unboxing
测试算子调用时的参数打包/解包:
- Boxed kernel (使用 `OperatorKernel::BoxedKernelFunction`)
- Unboxed kernel (直接调用 C++ 函数)
- 类型转换和性能优化

## 验证方式
- 使用 Google Test 框架(`TEST()`, `EXPECT_*`, `ASSERT_*`)
- 通过 `Dispatcher::singleton()` 获取全局调度器
- 调用 `findSchema()` 查找算子
- 使用 `callOp()` 或 `callOpUnboxed()` 执行算子
- 验证返回值、异常、状态变化

---

**ROCm 相关内容:**
- `DispatchKey::HIP` - ROCm GPU 后端的分发键

**Backward 相关内容:**
- `DispatchKey::AutogradCPU` / `AutogradCUDA` - 自动微分后端
- Autograd kernel 注册测试
- 梯度计算相关的算子实现注册
