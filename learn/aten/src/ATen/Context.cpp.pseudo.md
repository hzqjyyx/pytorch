```
--------------------------------
Context (全局配置与状态管理)
来源: aten/src/ATen/Context.cpp
--------------------------------

【状态】
    enabled_cudnn: bool                         # CuDNN 启用状态
    enabled_mkldnn: bool                        # MKLDNN 启用状态
    enabled_nnpack: bool                        # NNPACK 启用状态
    deterministic_cudnn: bool                   # CuDNN 确定性模式
    deterministic_mkldnn: bool                  # MKLDNN 确定性模式
    _deterministic_algorithms: bool             # 全局确定性算法开关
    _deterministic_algorithms_warn_only: bool   # 确定性违反时仅警告不报错
    _deterministic_fill_uninitialized_memory: bool  # 确定性填充未初始化内存
    benchmark_cudnn: bool                       # CuDNN benchmark 模式
    benchmark_limit_cudnn: int                  # CuDNN benchmark 限制
    allow_tf32_cudnn: bool                      # CuDNN TF32 加速
    allow_tf32_onednn: bool                     # OneDNN TF32 加速
    enabled_flashSDP: bool                      # Flash SDP 启用状态
    enabled_mem_efficientSDP: bool              # Memory Efficient SDP 启用状态
    enabled_mathSDP: bool                       # Math SDP 启用状态
    allow_fp16_bf16_reduction_mathSDP: bool     # Math SDP 允许 FP16/BF16 归约
    enabled_cudnnSDP: bool                      # CuDNN SDP 启用状态
    enabled_overrideable: bool                  # Overrideable SDP 启用状态
    sdp_priority_order: array[SDPBackend, num_sdp_backends]  # SDP 后端优先级顺序
    float32_matmul_precision: Float32MatmulPrecision  # FP32 矩阵乘精度 (HIGHEST|HIGH|MEDIUM)
    linalg_preferred_backend: LinalgBackend     # 线性代数首选后端
    blas_preferred_backend: BlasBackend         # BLAS 首选后端
    rocm_fa_preferred_backend: ROCmFABackend    # ROCm Flash Attention 首选后端
    allow_fp16_reduction_cublas: bool           # CuBLAS 允许 FP16 归约
    allow_bf16_reduction_cublas: bool           # CuBLAS 允许 BF16 归约
    allow_fp16_accumulation_cublas: bool        # CuBLAS 允许 FP16 累加
    allow_fp16_reduction_cpu: bool              # CPU 允许 FP16 归约
    sm_carveout: optional[int32]                # SM carveout 实验性参数
    quantized_engine: optional[QEngine]         # 量化引擎
    enable_sparse_tensor_invariant_checks: bool # 稀疏张量不变性检查
    release_original_weights: bool              # 预打包时释放原始权重
    prev_allocator_ptr_: Allocator*             # 前一个 CPU 分配器指针
    display_vmap_fallback_warnings_: bool       # 显示 vmap fallback 警告

【接口】
    # ========== 全局单例 ==========
    globalContext() -> Context&
        用途: 获取全局 Context 单例，静态懒初始化

    # ========== CuDNN 配置 ==========
    userEnabledCuDNN() -> bool
        用途: 查询用户是否启用 CuDNN
    
    setUserEnabledCuDNN(e: bool)
        用途: 设置 CuDNN 启用状态

    deterministicCuDNN() -> bool
        用途: 查询 CuDNN 确定性模式
    
    setDeterministicCuDNN(b: bool)
        用途: 设置 CuDNN 确定性模式

    benchmarkCuDNN() -> bool
        用途: 查询 CuDNN benchmark 模式
    
    setBenchmarkCuDNN(b: bool)
        用途: 设置 CuDNN benchmark 模式

    benchmarkLimitCuDNN() -> int
        用途: 查询 CuDNN benchmark 限制
    
    setBenchmarkLimitCuDNN(b: int)
        用途: 设置 CuDNN benchmark 限制

    allowTF32CuDNN() -> bool
        用途: 查询是否允许 CuDNN TF32 加速
    
    setAllowTF32CuDNN(b: bool)
        用途: 设置 CuDNN TF32 加速

    # ========== MKLDNN 配置 ==========
    userEnabledMkldnn() -> bool
        用途: 查询用户是否启用 MKLDNN
    
    setUserEnabledMkldnn(e: bool)
        用途: 设置 MKLDNN 启用状态

    deterministicMkldnn() -> bool
        用途: 查询 MKLDNN 确定性模式
    
    setDeterministicMkldnn(b: bool)
        用途: 设置 MKLDNN 确定性模式

    # ========== NNPACK 配置 ==========
    userEnabledNNPACK() -> bool
        用途: 查询用户是否启用 NNPACK
    
    setUserEnabledNNPACK(e: bool)
        用途: 设置 NNPACK 启用状态

    # ========== 确定性算法 ==========
    deterministicAlgorithms() -> bool
        用途: 查询全局确定性算法是否启用
    
    deterministicAlgorithmsWarnOnly() -> bool
        用途: 查询确定性违反时是否仅警告
    
    setDeterministicAlgorithms(b: bool, warn_only: bool = false)
        用途: 设置全局确定性算法策略

    deterministicFillUninitializedMemory() -> bool
        用途: 查询是否确定性填充未初始化内存
    
    setDeterministicFillUninitializedMemory(b: bool)
        用途: 设置确定性填充未初始化内存

    alertNotDeterministic(caller: string_view)
        用途: 警告或报错某操作不支持确定性
        调用: TORCH_WARN, TORCH_CHECK

    checkCuBLASConfigDeterministic() -> bool
        用途: 检查 CuBLAS 工作空间配置是否满足确定性要求
        调用: c10::utils::get_env(CUBLAS_WORKSPACE_CONFIG)
        备注: 检查环境变量是否为 ":4096:8" 或 ":16:8"

    alertCuBLASConfigNotDeterministic()
        用途: 警告或报错 CuBLAS 配置不满足确定性要求
        调用: TORCH_WARN, TORCH_CHECK

    # ========== SDP (Scaled Dot Product) 配置 ==========
    setSDPPriorityOrder(order: vector[int64]) 
        用途: 设置 SDP 后端优先级顺序
        调用: TORCH_CHECK

    sDPPriorityOrder() -> array[SDPBackend, num_sdp_backends]
        用途: 获取 SDP 后端优先级顺序

    userEnabledFlashSDP() -> bool
        用途: 查询 Flash SDP 启用状态
    
    setSDPUseFlash(e: bool)
        用途: 设置 Flash SDP 启用状态

    userEnabledMemEfficientSDP() -> bool
        用途: 查询 Memory Efficient SDP 启用状态
    
    setSDPUseMemEfficient(e: bool)
        用途: 设置 Memory Efficient SDP 启用状态

    userEnabledMathSDP() -> bool
        用途: 查询 Math SDP 启用状态
    
    setSDPUseMath(e: bool)
        用途: 设置 Math SDP 启用状态

    allowFP16BF16ReductionMathSDP() -> bool
        用途: 查询 Math SDP 是否允许 FP16/BF16 归约
    
    setAllowFP16BF16ReductionMathSDP(e: bool)
        用途: 设置 Math SDP FP16/BF16 归约

    userEnabledCuDNNSDP() -> bool
        用途: 查询 CuDNN SDP 启用状态
    
    setSDPUseCuDNN(e: bool)
        用途: 设置 CuDNN SDP 启用状态

    userEnabledOverrideableSDP() -> bool
        用途: 查询 Overrideable SDP 启用状态
    
    setSDPUseOverrideable(e: bool)
        用途: 设置 Overrideable SDP 启用状态

    # ========== TF32 配置 ==========
    allowTF32OneDNN() -> bool
        用途: 查询是否允许 OneDNN TF32 加速
    
    setAllowTF32OneDNN(b: bool)
        用途: 设置 OneDNN TF32 加速 (仅 XPU 平台有效)
        调用: TORCH_WARN

    allowTF32CuBLAS() -> bool
        用途: 查询是否允许 CuBLAS TF32 加速
        备注: 通过 float32_matmul_precision != HIGHEST 判断
    
    setAllowTF32CuBLAS(b: bool)
        用途: 设置 CuBLAS TF32 加速
        备注: 实际修改 float32_matmul_precision 为 HIGH/HIGHEST

    # ========== FP32 矩阵乘精度 ==========
    float32MatmulPrecision() -> Float32MatmulPrecision
        用途: 获取 FP32 矩阵乘精度

    setFloat32MatmulPrecision(p: Float32MatmulPrecision)
        用途: 设置 FP32 矩阵乘精度 (枚举值)

    setFloat32MatmulPrecision(s: string)
        参数: s = "highest" | "high" | "medium" (大小写不敏感)
        用途: 通过字符串设置 FP32 矩阵乘精度
        调用: TORCH_WARN

    # ========== 线性代数后端 ==========
    linalgPreferredBackend() -> LinalgBackend
        用途: 获取线性代数首选后端

    setLinalgPreferredBackend(b: LinalgBackend)
        用途: 设置线性代数首选后端
        调用: TORCH_CHECK, hasCuSOLVER(), hasMAGMA(), TORCH_WARN_ONCE

    blasPreferredBackend() -> BlasBackend
        用途: 获取 BLAS 首选后端，包含架构检测逻辑
        调用: detail::getCUDAHooks().deviceCount(), detail::getCUDAHooks().isGPUArch()
        备注: Default 会根据平台自动转换为 Cublas 或 Cublaslt

    setBlasPreferredBackend(b: BlasBackend)
        用途: 设置 BLAS 首选后端
        调用: TORCH_CHECK, TORCH_WARN_ONCE, hasCuBLASLt(), hasROCM()

    getROCmFAPreferredBackend() -> ROCmFABackend
        用途: 获取 ROCm Flash Attention 首选后端

    setROCmFAPreferredBackend(b: ROCmFABackend)
        用途: 设置 ROCm Flash Attention 首选后端
        调用: TORCH_CHECK, TORCH_WARN_ONCE, detail::getCUDAHooks()

    # ========== CuBLAS 精度配置 ==========
    allowFP16ReductionCuBLAS() -> bool
        用途: 查询 CuBLAS 是否允许 FP16 归约

    setAllowFP16ReductionCuBLAS(b: bool)
        用途: 设置 CuBLAS FP16 归约

    allowBF16ReductionCuBLAS() -> bool
        用途: 查询 CuBLAS 是否允许 BF16 归约

    setAllowBF16ReductionCuBLAS(b: bool)
        用途: 设置 CuBLAS BF16 归约

    allowFP16AccumulationCuBLAS() -> bool
        用途: 查询 CuBLAS 是否允许 FP16 累加

    setAllowFP16AccumulationCuBLAS(b: bool)
        用途: 设置 CuBLAS FP16 累加

    # ========== CPU 精度配置 ==========
    allowFP16ReductionCPU() -> bool
        用途: 查询 CPU 是否允许 FP16 归约

    setAllowFP16ReductionCPU(b: bool)
        用途: 设置 CPU FP16 归约
        调用: cpuinfo_initialize(), cpuinfo_has_arm_fp16_arith()
        备注: 仅 aarch64 平台支持，需检查硬件能力

    # ========== 实验性功能 ==========
    _SMCarveout_EXPERIMENTAL() -> optional[int32]
        用途: 获取 SM carveout 实验性参数

    _setSMCarveout_EXPERIMENTAL(c: optional[int32])
        用途: 设置 SM carveout 实验性参数
        调用: TORCH_WARN_ONCE

    # ========== 库支持查询 ==========
    hasMKL() -> bool
        用途: 查询是否编译了 MKL 支持

    hasMKLDNN() -> bool
        用途: 查询是否编译了 MKLDNN 支持

    hasKleidiAI() -> bool
        用途: 查询是否编译了 KleidiAI 支持
        调用: AT_KLEIDIAI_ENABLED()

    hasOpenMP() -> bool
        用途: 查询是否编译了 OpenMP 支持

    hasLAPACK() -> bool
        用途: 查询是否编译了 LAPACK 支持

    isXNNPACKAvailable() -> bool
        用途: 查询是否编译了 XNNPACK 支持

    # ========== 量化引擎 ==========
    qEngine() -> QEngine
        用途: 获取当前量化引擎，含平台默认值逻辑
        调用: fbgemm::fbgemmSupportedCPU()
        备注: 优先级 NoQEngine < QNNPACK < ONEDNN < X86/FBGEMM

    setQEngine(e: QEngine)
        用途: 设置量化引擎
        调用: TORCH_CHECK, supportedQEngines()

    supportedQEngines() -> vector[QEngine]
        用途: 获取支持的量化引擎列表
        调用: fbgemm::fbgemmSupportedCPU()

    # ========== 稀疏张量 ==========
    checkSparseTensorInvariants() -> bool
        用途: 查询是否启用稀疏张量不变性检查

    setCheckSparseTensorInvariants(e: bool)
        用途: 设置稀疏张量不变性检查

    # ========== 权重预打包 ==========
    releaseWeightsWhenPrepacking() -> bool
        用途: 查询预打包时是否释放原始权重

    setReleaseWeightsWhenPrepacking(e: bool)
        用途: 设置预打包时释放原始权重

    # ========== CPU 优化 ==========
    setFlushDenormal(on: bool) -> bool
        用途: 设置 CPU flush-to-zero 模式
        调用: at::cpu::set_flush_denormal()

    # ========== 内存分配器 ==========
    isDefaultMobileCPUAllocatorSet() -> bool
        用途: 查询是否已设置移动端 CPU 分配器

    setDefaultMobileCPUAllocator()
        用途: 切换到移动端优化的 CPU 分配器
        调用: TORCH_CHECK, c10::GetCPUAllocator(), c10::SetCPUAllocator(), c10::GetDefaultMobileCPUAllocator()

    unsetDefaultMobileCPUAllocator()
        用途: 恢复原始 CPU 分配器
        调用: TORCH_CHECK, c10::SetCPUAllocator()

    # ========== Vmap 警告 ==========
    areVmapFallbackWarningsEnabled() -> bool
        用途: 查询是否启用 vmap fallback 警告

    setDisplayVmapFallbackWarnings(enabled: bool)
        用途: 设置 vmap fallback 警告

【依赖】
    TORCH_CHECK / TORCH_WARN / TORCH_WARN_ONCE    # 错误检查与警告
    c10::utils::get_env / check_env                # 环境变量读取
    c10::GetCPUAllocator / SetCPUAllocator        # CPU 分配器管理
    detail::getCUDAHooks()                         # CUDA 设备查询

【平台特定】
    ROCm:
        - hipBLASLt: 需检查 HIPBLASLT_ALLOW_TF32 环境变量
        - hipBLASLt 架构支持: gfx90a, gfx942, gfx950 (6.5+), gfx1100/1101/1200/1201 (6.3+)
        - CK 架构支持: gfx90a, gfx942
    
    CUDA:
        - CuBLAS 确定性: 需设置 CUBLAS_WORKSPACE_CONFIG 为 ":4096:8" 或 ":16:8"
    
    CPU:
        - FP16 归约: 仅 aarch64 非移动平台支持，需检查 cpuinfo_has_arm_fp16_arith()

--------------------------------
Thread-Local Guards
来源: aten/src/ATen/Context.cpp
--------------------------------

【Thread-Local 状态】
    override_allow_tf32_flag: bool    # TF32 禁用标志
    rocm_is_backward_pass: bool       # ROCm 反向传播标志

【Guard 类】
    NoTF32Guard:
        用途: RAII guard，在作用域内强制禁用 TF32
        构造: 设置 override_allow_tf32_flag = true
        析构: 恢复 override_allow_tf32_flag = false
        静态方法: should_disable_tf32() -> bool

    ROCmBackwardPassGuard:
        用途: RAII guard，标记反向传播阶段
        构造: 设置 rocm_is_backward_pass = true
        析构: 恢复 rocm_is_backward_pass = false
        静态方法: is_backward_pass() -> bool

--------------------------------
自由函数
来源: aten/src/ATen/Context.cpp
--------------------------------

【函数】
    getCPUAllocator() -> Allocator*
        用途: 获取全局 CPU 内存分配器
        调用: c10::GetCPUAllocator()
```
