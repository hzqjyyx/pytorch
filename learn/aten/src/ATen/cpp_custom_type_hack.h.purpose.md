这个文件提供了一个**已废弃的临时机制**，用于将任意 C++ 对象嵌入到 PyTorch Tensor 中。

## 核心机制

通过滥用 Tensor 的存储系统来存储自定义 C++ 对象：

1. **创建 (`create`)** - `aten/src/ATen/cpp_custom_type_hack.h:91`
   - 接收 `std::unique_ptr<T>` 和 TensorOptions
   - 将原始指针包装进 `at::DataPtr`，注册类型 T 的删除器
   - 创建一个字节类型的空 Tensor（大小为 `sizeof(T)`）
   - 将自定义对象的指针替换到 Tensor 的存储中
   - 禁用追踪和自动微分

2. **类型检查 (`isa`)** - `aten/src/ATen/cpp_custom_type_hack.h:66`
   - 检查 Tensor 是否为 kByte 类型
   - 通过比较存储的删除器函数指针来验证类型

3. **类型转换 (`cast`)** - `aten/src/ATen/cpp_custom_type_hack.h:76`
   - 验证 Tensor 类型和删除器
   - 通过 `reinterpret_cast` 将存储指针转换回 T* 类型

## 为什么不安全

- 绕过了 PyTorch 的类型系统
- 依赖未定义的内存布局假设
- 没有序列化/反序列化支持
- 与 JIT 编译器不兼容

## 替代方案

文件明确指出应使用 **custom classes** 机制：
https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html

所有三个函数都标记为 `[[deprecated]]`，警告不要添加新的调用点。

---

**忽略内容：**
- ROCm 相关：无
- Backward 相关：无（文件中禁用了自动微分追踪）
