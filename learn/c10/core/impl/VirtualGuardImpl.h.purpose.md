## VirtualGuardImpl.h 文件分析

这个文件定义了 `VirtualGuardImpl` 类，它是一个**虚拟设备守卫的实现**，通过动态分派模式将设备操作委托给具体的设备实现。

### 核心设计

`VirtualGuardImpl` 继承自 `DeviceGuardImplInterface`，内部持有一个指向具体设备守卫实现的指针 `impl_`。所有的设备操作（如获取设备、设置设备、流管理等）都通过这个指针转发给真实的实现。

### 主要功能

- **设备管理**：类型查询、设备交换、获取/设置当前设备
- **流管理**：获取/创建/交换计算流，支持优先级和全局流池
- **事件管理**：记录、查询、同步和销毁 GPU 事件，计算事件间的时间差
- **同步操作**：同步流和设备，支持将数据指针绑定到流
- **设备查询**：获取设备总数

### 关键特性

- 通过虚拟继承实现**设备无关的接口**，支持多种设备后端
- 提供了两个构造函数：一个基于 `DeviceType` 自动查询实现，一个用于测试直接注入实现
- 支持**复制和移动语义**，默认实现效率高

### 主要方法列表

- `type()` - 返回设备类型
- `exchangeDevice(d)` / `setDevice(d)` / `getDevice()` - 设备切换和查询
- `getStream()` / `getNewStream()` / `getDefaultStream()` / `getStreamFromGlobalPool()` - 流管理
- `record()` / `block()` / `queryEvent()` / `synchronizeEvent()` / `destroyEvent()` - 事件处理
- `queryStream()` / `synchronizeStream()` - 流同步
- `recordDataPtrOnStream()` - 数据指针流绑定
- `elapsedTime()` - 事件间时间计算
- `synchronizeDevice()` - 设备同步
