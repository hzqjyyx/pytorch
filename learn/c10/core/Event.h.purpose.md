## Event.h 主要功能解析

这个文件定义了 PyTorch c10 库的通用事件（Event）类，用于跨后端的设备同步机制。

**核心设计**：
- 后端通用的可移动事件（movable but not copyable/thread-safe）
- 基于 CUDA/HIP 事件设计模式
- 建立在 DeviceGuardImpl 基础上，支持任意编译时未知的后端

**事件版本机制**：
- 支持重复记录（rerecord），每次创建新版本
- 等待线程只会等待记录时的版本，不会受后续重录影响
- 查询只返回最新版本的状态

**主要功能**：

- **构造/析构**：删除拷贝，支持移动语义
- **设备信息获取**：device()、device_type()、device_index()、flag()
- **事件记录**：
  - `record()`：增加版本号，在流中入队录制任务
  - `recordOnce()`：仅首次调用时记录
- **流同步**：
  - `block()`：阻塞流直到事件版本完成
  - `synchronize()`：主机端等待事件完成
- **事件查询**：
  - `query()`：检查是否未记录或已完成
  - `elapsedTime()`：计算两事件间耗时
  - `eventId()`：获取事件指针
- **状态检查**：was_marked_for_recording()

**实现细节**：
- 内部使用 `impl::InlineEvent<impl::VirtualGuardImpl>` 存储后端实现
- 通过虚拟守卫接口适配不同设备后端
