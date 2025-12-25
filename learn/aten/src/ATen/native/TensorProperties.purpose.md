## TensorProperties.h

这个头文件是一个简单的接口定义文件，只导出了一个函数声明：

- `cudnn_is_acceptable(const TensorBase& self)` - 检查张量是否可被 cuDNN 库接受

## TensorProperties.cpp

这个文件实现了张量属性查询和操作的一系列功能：

### 主要功能分组

**1. 张量尺寸比较**
- `is_same_size()` - 比较两个张量的符号尺寸是否相同
- `nested_is_same_size()` - 比较两个嵌套张量的尺寸是否相同

**2. 张量维度查询**
- `size(dim)` - 返回指定维度的大小（int64_t）
- `stride(dim)` - 返回指定维度的步长
- 支持按整数维度或命名维度（Dimname）查询

**3. 符号整数操作**
- `sym_size()` - 返回符号整数形式的维度大小
- `sym_stride()` - 返回符号整数形式的步长
- `sym_numel()` - 返回符号整数形式的总元素数
- `sym_storage_offset()` - 返回符号整数形式的存储偏移

**4. cuDNN 兼容性检查**
- `cudnn_is_acceptable()` - 验证张量是否满足 cuDNN 使用条件：
  - cuDNN 已启用
  - 张量在 CUDA 设备上
  - 数据类型为 double、float 或 half
  - cuDNN 库已编译支持
  - 张量非空（numel > 0）

**5. 张量内存管理**
- `contiguous()` - 返回按指定内存格式连续的张量副本
- `detach_()` - 分离张量（钩子方法）
- `is_set_to()` - 检查两个张量是否共享相同的存储和布局

## 核心特点

- **符号整数支持** - 使用 `SymInt` 支持动态形状
- **嵌套张量支持** - 特殊处理嵌套张量的操作
- **设备感知** - 针对 CUDA 和 cuDNN 的特定检查
- **命名维度** - 支持通过维度名称查询属性
