GeneratorImpl 是 PyTorch 中伪随机数生成器(PRNG)的核心实现类，主要用于管理随机数生成的状态和种子。

## 核心功能

**基类设计**
- GeneratorImpl 继承自 `intrusive_ptr_target`，支持引用计数管理
- 删除了拷贝和移动构造/赋值，仅支持通过 `clone()` 方法进行深拷贝

**状态管理**
- 存储设备信息(CPU/CUDA 等)和 DispatchKeySet(用于分发调用)
- 维护 `std::mutex` 用于线程安全(注:文档说明目前线程安全性还不完善)
- 保持 PyObject 指针用于 Python 绑定

**虚拟接口**
- `set_current_seed()` / `current_seed()` - 种子管理
- `set_offset()` / `get_offset()` - 随机数序列偏移
- `seed()` - 生成新的随机种子
- `set_state()` / `get_state()` - 状态序列化/反序列化
- `graphsafe_set_state()` / `graphsafe_get_state()` - 图模式下的状态管理

**随机数生成辅助**
- `getNonDeterministicRandom()` - 获取非确定性随机数
  - Windows: 使用 `std::chrono::high_resolution_clock`
  - SGX: 使用 `sgx_read_rand`
  - Unix/Linux: 读取 `/dev/urandom`
  - CUDA: 使用 `std::random_device` 并限制在 53 位以确保双精度浮点表示唯一性

**核心设计特点**
- 每个设备默认有一个生成器，延迟创建
- 支持多个用户定义的生成器实例
- 通过虚拟方法模式支持不同后端(CPU/CUDA/等)的具体实现

---

• **用途**: 伪随机数生成器的基础框架
• **核心**: 虚拟方法接口 + 设备和种子状态管理
• **线程**: 提供 mutex_ 但线程安全性待完善
• **克隆**: 支持生成器深拷贝，禁用拷贝赋值
• **随机源**: 跨平台适配(/dev/urandom、时钟、SGX、CUDA)
• **扩展**: 子类实现具体算法(如 CPUGeneratorImpl)
