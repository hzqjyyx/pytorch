## XPU 设备管理核心功能

### 设备枚举与初始化

**设备发现策略** (`enumDevices`)：
- 遍历所有 SYCL 平台，筛选 Level Zero 后端
- 优先枚举离散 GPU (dGPU)，其次集成 GPU (iGPU)
- 通过 `host_unified_memory` 特性区分两者
- 在 Windows 上同一平台内的 GPU 共享上下文，跨平台 GPU 各自创建上下文

**全局状态管理**：
- `gDevicePool`：存储枚举得到的设备列表和共享上下文
- `curDeviceIndex`：线程本地当前活跃设备索引
- `init_flag`：确保设备初始化只执行一次（`call_once` 机制）

### 公开 API

**设备计数与切换**：
- `device_count()`：返回设备总数（无设备时警告）
- `device_count_ensure_non_zero()`：确保至少有一个设备（无则错误）
- `current_device()`：获取当前活跃设备索引
- `set_device()`：切换活跃设备
- `exchange_device()` / `maybe_exchange_device()`：原子性设备切换

**底层访问**：
- `get_raw_device()`：获取指定设备的 SYCL device 对象
- `get_device_context()`：获取全局 SYCL 上下文
- `get_device_properties()`：查询设备属性（计算能力、内存等）
- `get_device_idx_from_pointer()`：从 USM 指针反查所属设备

### 关键设计细节

- **宏驱动属性初始化**：`AT_FORALL_XPU_DEVICE_PROPERTIES` 等宏批量赋值设备属性
- **错误处理**：驱动异常时警告而非崩溃，保障用户体验
- **平台差异处理**：Windows 前 2025.0.0 编译器版本需手动构建上下文

---

### 功能总结

- 🔍 发现并枚举 Intel GPU 设备（SYCL 运行时）
- 🎯 提供设备选择与切换机制
- 📊 查询设备属性与 USM 内存归属
- 🔒 线程安全的全局设备池管理
- ⚙️ 适配 Windows/Linux 平台与 dGPU/iGPU 差异
