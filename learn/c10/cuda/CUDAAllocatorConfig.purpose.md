# CUDAAllocatorConfig 核心功能

## 配置管理器
这是 PyTorch CUDA 缓存分配器的配置解析和管理系统，通过环境变量 `PYTORCH_CUDA_ALLOC_CONF` 控制内存分配行为。

## 主要配置项

### 1. **max_split_size_mb**
- 控制内存块分割的最大尺寸
- 默认：无限制 (`std::numeric_limits<size_t>::max()`)
- 解析位置：`parseMaxSplitSize()` (c10/cuda/CUDAAllocatorConfig.cpp:77-96)

### 2. **max_non_split_rounding_mb**
- 设置不进行分割的内存块 rounding 尺寸上限
- 默认：`kLargeBuffer`
- 解析位置：`parseMaxNonSplitRoundingSize()` (c10/cuda/CUDAAllocatorConfig.cpp:98-117)

### 3. **roundup_power2_divisions**
- 将分配尺寸向上取整到 2 的幂次的除数
- 覆盖范围：1MB (2^20) 到 64GB (2^36)，共 16 个区间
- 支持两种配置格式：
  - 单一值：`roundup_power2_divisions:4` - 全部区间使用同一除数
  - 区间映射：`roundup_power2_divisions:[1048576:2,8388608:4,>:0]` - 不同尺寸区间使用不同除数
- 解析逻辑：`parseRoundUpPower2Divisions()` (c10/cuda/CUDAAllocatorConfig.cpp:137-217)
- 查询接口：`roundup_power2_divisions(size_t size)` - 根据分配尺寸返回对应除数 (c10/cuda/CUDAAllocatorConfig.cpp:25-42)

### 4. **garbage_collection_threshold**
- 垃圾回收触发阈值（0.0-1.0）
- 默认：0（禁用）
- 解析位置：`parseGarbageCollectionThreshold()` (c10/cuda/CUDAAllocatorConfig.cpp:119-135)

### 5. **expandable_segments**
- 启用可扩展内存段
- 需要 `PYTORCH_C10_DRIVER_API_SUPPORTED` 支持
- 默认：false
- 解析位置：`parseArgs()` (c10/cuda/CUDAAllocatorConfig.cpp:295-305)

### 6. **backend**
- 分配器后端选择：`native` 或 `cudaMallocAsync`
- cudaMallocAsync 需要 CUDA 11.4+
- 解析位置：`parseAllocatorConfig()` (c10/cuda/CUDAAllocatorConfig.cpp:219-258)

### 7. **Pinned Memory 配置**
- `pinned_use_cuda_host_register`：使用 cudaHostRegister (默认: false)
- `pinned_num_register_threads`：注册线程数，必须是 2 的幂，上限 128 (默认: 1)
- `pinned_use_background_threads`：使用后台线程 (默认: false)

## 核心机制

### 配置解析流程
1. **词法分析** (`lexArgs()` - c10/cuda/CUDAAllocatorConfig.cpp:44-64)
   - 将环境变量字符串分割为 tokens
   - 分隔符：`,` `:` `[` `]`
   - 自动过滤空格

2. **语法解析** (`parseArgs()` - c10/cuda/CUDAAllocatorConfig.cpp:260-353)
   - 遍历 tokens，识别配置项名称
   - 调用对应的 `parse*()` 方法解析值
   - 验证 native-specific 选项与 cudaMallocAsync 的冲突

3. **Token 消费** (`consumeToken()` - c10/cuda/CUDAAllocatorConfig.cpp:66-75)
   - 验证预期的分隔符是否存在
   - 失败时抛出 TORCH_CHECK 异常

### 单例模式
- 静态初始化：`instance()` (c10/cuda/CUDAAllocatorConfig.h:80-88)
- 首次调用时读取 `PYTORCH_CUDA_ALLOC_CONF` 环境变量
- 全局唯一配置实例

### 线程安全
- `m_last_allocator_settings` 使用 `std::mutex` 保护 (c10/cuda/CUDAAllocatorConfig.cpp:272-274)
- 大部分配置项使用 `std::atomic` 类型 (c10/cuda/CUDAAllocatorConfig.h:124-132)

## 使用示例

```bash
# 组合配置
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8,roundup_power2_divisions:[1048576:2,8388608:4,>:0]"

# 启用可扩展段
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 使用 cudaMallocAsync 后端
PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
```

## 外部接口
- `setAllocatorSettings(const std::string& env)` (c10/cuda/CUDAAllocatorConfig.cpp:412-414) - 运行时修改配置

---

**其他内容：**
- ROCm 兼容：通过 `hipify` 将 CUDA API 转换为 HIP，配置项名称同时接受 `hip` 和 `cuda` 前缀
- Backward compatibility：`roundup_power2_divisions` 保留旧的单一值格式支持 (c10/cuda/CUDAAllocatorConfig.cpp:202-212)
