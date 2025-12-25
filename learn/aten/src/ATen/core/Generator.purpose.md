Generator 是 PyTorch 中的伪随机数生成器（PRNG）接口类，通过 PIMPL 模式封装了具体实现。

**主要功能：**

• **状态管理** - `set_state()`/`get_state()` 用于保存和恢复 PRNG 状态，支持序列化和复现随机结果

• **种子控制** - `set_current_seed()` 和 `seed()` 用于设置和查询生成器的种子值

• **偏移量设置** - `set_offset()`/`get_offset()` 用于 Philox 算法（CUDA/MPS）调整状态偏移

• **计算图安全** - `graphsafe_set_state()`/`graphsafe_get_state()` 用于在动态计算图中安全地操作生成器状态

• **设备绑定** - 每个设备（CPU/CUDA/MPS）有对应的生成器实现，`device()` 返回生成器所属设备

• **线程同步** - 提供 `mutex()` 接口用于多线程环境下的锁保护（文件注释说明生成器本身非线程安全）

• **克隆和转换** - `clone()` 创建副本，`get<T>()` 用于强制类型转换到具体后端实现类

• **工厂函数** - `make_generator()` 模板函数用于创建特定实现的生成器，`check_generator()` 和 `get_generator_or_default()` 用于验证和获取生成器

• **Python 绑定** - `set_pyobj()`/`pyobj()` 用于关联 Python 对象
