## Stream 类的主要功能

**Stream** 是 PyTorch 中用于同步 GPU 内核执行的软件机制，是一个设备无关的值类。

### 核心设计理念

- 每个内核启动都关联一个 stream
- 同一个 stream 上的内核隐式同步（按顺序执行）
- 不同 stream 上的内核可以并发执行
- 每个 stream 都绑定到特定的 Device
- Stream 是线程安全的，可以在线程间传递

### 类结构

**私有成员：**
- `Device device_` - 关联的设备
- `StreamId id_` - 流的 ID（int64_t）

**构造方式：**
- `Stream(UNSAFE, device, id)` - 底层构造（仅内部使用）
- `Stream(DEFAULT, device)` - 构造默认 stream（id=0）

### 主要 API

**查询和同步：**
- `query()` - 检查 stream 上的异步工作是否完成（不阻塞）
- `synchronize()` - 阻塞调用线程直到 stream 完成
- `wait(event)` - 在 stream 中插入等待指令

**访问器：**
- `device()` / `device_type()` / `device_index()` - 获取设备信息
- `id()` - 获取 stream ID

**比较和哈希：**
- `operator==` / `operator!=` - 流的相等性比较
- `hash()` - 将 stream 打包为 uint64_t（便于 Python 绑定）
- `pack3()` / `unpack3()` - 拆包/打包为 StreamData3 结构

### 实现细节

- `Stream::query()` 和 `Stream::synchronize()` 委托给 `VirtualGuardImpl` 处理后端特定逻辑
- 支持 std::hash 特化，可用于 unordered_map 等容器

### 主要用途列表

- GPU 内核执行的队列化和同步
- 并发内核执行的控制
- Device 和 stream 的绑定管理
- Python 与 C++ 的流对象互操作
- 标准库容器（hash map）的支持
