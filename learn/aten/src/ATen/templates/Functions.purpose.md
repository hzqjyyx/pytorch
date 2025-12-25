## Functions.cpp - TensorMaker 实现

**TensorMaker::make_tensor()** 是核心方法，负责创建张量对象：

1. 禁用追踪和自动求导模式（AutoDispatchBelowADInplaceOrView、NoTracerDispatchMode）
2. 验证大小非负，deleter 和 context 互斥
3. 从指针推断或验证设备匹配
4. 计算存储大小（根据 strides 或连续存储）
5. 构造 DataPtr（从 deleter 或 context）
6. 创建 Storage 对象，标记是否可调整大小
7. 通过 detail::make_tensor 创建 TensorImpl
8. 配置张量的 sizes、strides、storage_offset

**computeStorageSize()** 计算字节大小：
- 使用 strides 时：调用 detail::computeStorageNbytes
- 连续存储时：直接计算 size × itemsize
- 加上 storage_offset（如有）

**数据指针生成**：
- makeDataPtrFromDeleter()：用自定义删除器包装
- makeDataPtrFromContext()：从智能指针上下文提取

**makeTempSizes()** 为 MemoryFormat 返回临时尺寸数组（用于通道优先格式）

---

## Functions.h - 张量 API 声明

- 包含头注释强调增量编译问题，推荐使用 `ATen/ops/{op}.h` 代替全局 `Functions.h`
- 包含必要依赖：Context、DeviceGuard、TensorUtils、Generator、Tensor、Storage、TensorOptions 等
- 声明所有 ATen 函数（通过 `${Functions_declarations}` 模板变量生成）
- 特殊 C++ 重载处理整数到 bool/IntArrayRef 的优先级问题（std、var、std_mean、var_mean）
- 内联辅助函数封装张量方法（numel、size、stride、is_complex 等）

---

**核心职责总结：**

- **Functions.cpp**: TensorMaker 类实现张量底层构造逻辑（内存分配、设备绑定、布局配置）
- **Functions.h**: C++ 张量算子接口声明和便捷包装函数
