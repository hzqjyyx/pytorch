# ATen TunableOp 主要功能分析

这是一个**自适应内核选择和性能优化系统**，允许 PyTorch 在运行时为相同操作的不同实现进行基准测试，并记录最优选择。

## 核心组件

### 1. TuningResultsManager (管理调优结果)
- **存储结构**：两层映射 `op_signature -> params_signature -> ResultEntry`
  - `op_signature`: 操作类型标识（如 "gemm", "conv"）
  - `params_signature`: 具体参数配置（如矩阵维度、数据类型）
  - `ResultEntry`: 最佳内核 ID + 执行时间
  
- **核心功能**：
  - `Lookup()`: 查询给定操作和参数的最优内核
  - `Add()`: 添加新的调优结果，拒绝覆盖已有结果（aten/src/ATen/cuda/tunable/Tunable.cpp:96-102）
  - `Load()/Dump()`: 批量导入/导出所有结果
  - `RecordUntuned()`: 记录未调优的操作配置到 CSV 文件

### 2. TuningResultsValidator (验证调优结果兼容性)
- **验证机制**：确保加载的调优结果与当前环境匹配
  - 必需验证项：`PT_VERSION`（PyTorch 版本）
  - 验证流程：检查强制键 -> 检查键匹配 -> 逐个验证值（aten/src/ATen/cuda/tunable/Tunable.cpp:345-371）

- **扩展点**：通过 `RegisterValidator()` 添加自定义验证器

### 3. TuningContext (全局配置管理单例)
获取方式：`getTuningContext()` 返回静态实例（aten/src/ATen/cuda/tunable/Tunable.cpp:47-50）

**启用控制**：
```cpp
// 三级开关设计
bool enable_;              // 总开关：是否启用 TunableOp
bool tuning_enable_;       // 是否允许运行时调优（false 时仅使用已有结果）
bool record_untuned_enable_; // 是否记录未调优的配置
```

**调优参数**：
- `max_tuning_duration_ms_`: 单次调优最长时间（默认 30ms）
- `max_tuning_iterations_`: 最多尝试多少个内核（默认 100）
- `max_warmup_*`: 预热阶段参数
- `icache_flush_`: 是否在测试间清空指令缓存（避免缓存影响）
- `rotating_buffer_size_`: 旋转缓冲区大小（默认 L2 缓存大小）

**文件持久化**：
- 自动在析构时保存新结果（aten/src/ATen/cuda/tunable/Tunable.cpp:425-436）
- 文件名支持 `%d` 占位符插入设备序号（避免多进程冲突）
- CSV 格式：`Validator,key,value` + `op_sig,param_sig,kernel_id,time`

**环境变量支持**（所有配置都可通过环境变量覆盖）：
- `PYTORCH_TUNABLEOP_ENABLED=1`: 强制启用
- `PYTORCH_TUNABLEOP_TUNING=0`: 禁用调优（仅查表）
- `PYTORCH_TUNABLEOP_FILENAME`: 指定结果文件路径
- `PYTORCH_TUNABLEOP_VERBOSE=N`: 日志级别（1-3）

**调试支持**：
- `TUNABLE_LOG1/2/3` 宏提供三级日志输出
- 日志可重定向到文件/stdout/stderr（aten/src/ATen/cuda/tunable/Tunable.cpp:781-791）
- `numerics_check_enable_`: 启用数值正确性检查

## 典型工作流程

```
┌──────────────────────────────┐
│ 1. 运算请求（如 GEMM M=1024）│
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 2. 生成签名 "gemm(1024,512)" │
└──────────┬───────────────────┘
           │
           ▼
      查表命中？
       /      \
     是         否
     │          │
     ▼          ▼
  使用缓存   启用调优？
  结果        /    \
            是      否
            │        │
            ▼        ▼
         测试所有  使用默认
         候选内核   实现
            │
            ▼
         选最快的
            │
            ▼
      保存到表 + 写文件
```

## 设计亮点

1. **懒加载初始化**：结果文件在首次访问 `GetTuningResultsManager()` 时才读取（aten/src/ATen/cuda/tunable/Tunable.cpp:630-648）
2. **线程安全**：所有 Manager 操作都用 `std::scoped_lock` 保护
3. **优雅降级**：环境不匹配时拒绝加载结果，避免使用错误的内核选择
4. **即时写入验证**：初始化时尝试打开文件写入，提前发现权限问题（aten/src/ATen/cuda/tunable/Tunable.cpp:642-645）
5. **增量保存**：仅当有新结果时才重写文件（aten/src/ATen/cuda/tunable/Tunable.cpp:426-427）

---

**ROCm 相关**：
- 验证 `ROCM_VERSION`, `GCN_ARCH_NAME`, `ROCBLAS_VERSION`, `HIPBLASLT_VERSION`（aten/src/ATen/cuda/tunable/Tunable.cpp:223-279）

**其他细节**：
- `ITimer` 接口：抽象的计时器接口，供子类实现具体计时逻辑（aten/src/ATen/cuda/tunable/Tunable.h:229-239）
- `PYTORCH_TUNABLEOP_BLAS_LOG=1`：在 CSV 中额外记录 BLAS 参数签名（aten/src/ATen/cuda/tunable/Tunable.cpp:53-59）
