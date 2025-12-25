这个文件定义了 `RRefInterface` 抽象基类，是 PyTorch 分布式通信的核心接口。

**主要功能：**

- **RRef 的抽象接口** - 提供统一的远程引用对象定义，供 JIT 和分布式模块共用
- **所有权查询** - `owner()` 返回所有者的 worker ID，`ownerName()` 返回所有者名称
- **角色判断** - `isOwner()` 判断是否为 OwnerRRef，`confirmedByOwner()` 判断是否被所有者确认
- **类型信息** - `type()` 返回引用对象的类型信息
- **引用计数管理** - 继承自 `intrusive_ptr_target`，支持智能指针自动管理生命周期
- **非复制非移动** - 禁用拷贝和移动构造，防止引用计数混乱
