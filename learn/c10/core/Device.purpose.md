## Device.h 核心结构

定义了 `Device` 类，用于表示张量所在的计算设备：

- **设备类型 (DeviceType)**：CPU、CUDA、IPU、XPU、HIP 等多种后端
- **设备索引 (DeviceIndex)**：int8_t 类型，用于标识同类型设备中的具体实例（如第几个GPU）
  - `-1` 表示当前设备（默认值）
  - CPU 设备的索引必须为 0 或 -1

**主要接口**：

- `Device(DeviceType type, DeviceIndex index = -1)` - 通过类型和索引构造
- `Device(const std::string& device_string)` - 通过字符串构造（如 "cuda:1"）
- `type()`、`index()` - 获取设备信息
- `has_index()` - 判断是否指定了具体索引
- `is_cuda()`、`is_cpu()`、`is_mps()` 等 - 设备类型判断方法
- `str()` - 返回字符串表示

**哈希支持**：提供了 `std::hash<c10::Device>` 专门化，方便用作 map/set 的 key

---

## Device.cpp 核心逻辑

**1. 字符串解析 `parse_type()`**
- 将设备名称字符串（如 "cuda"、"cpu"）映射到 `DeviceType` 枚举值
- 支持19种设备类型，包括已废弃的 "mkldnn"（会触发警告）
- 支持自定义后端通过 `get_privateuse1_backend()` 注册

**2. 设备字符串构造函数**
- 使用状态机解析设备字符串，遵循正则表达式：`([a-zA-Z_]+)(?::([1-9]\d*|0))?`
- 格式：`<device_name>[:<device_index>]`，如 "cuda:0"、"cpu"
- 状态转移：START → INDEX_START → INDEX_REST，检查非零前导位和数字有效性
- 解析失败或数字溢出时抛出异常

**3. 输出操作**
- `str()` - 返回格式化字符串，包含设备类型（小写）和可选的索引
- `operator<<` - 支持流输出

---

## 关键特性总结

- **多后端支持**：19种设备类型，灵活的设备索引机制
- **字符串互转**：双向转换设备对象与字符串描述
- **验证机制**：构造时验证设备索引的有效性（DEBUG 模式）
- **哈希支持**：通过位操作组合类型和索引，避免符号扩展问题
- **状态机解析**：健壮的字符串解析，拒绝无效格式
