## Registry.h 文件分析

这是 PyTorch 的 C10 库中的一个**通用注册表实现**，用于在程序初始化时动态注册对象创建器。

### 核心机制

**Registry 模板类** (第 54-167 行)
- 存储键值对，其中键是标识符（通常为 std::string），值是创建对象的函数指针
- 支持三级优先级系统：FALLBACK(1) → DEFAULT(2) → PREFERRED(3)
- 高优先级注册可以覆盖低优先级注册，同优先级重复注册会报错
- 线程安全：使用 std::mutex 保护并发访问

**Registerer 助手类** (第 169-193 行)
- 简化注册过程，提供 DefaultCreator 模板方法自动生成创建器

### 主要操作

| 方法 | 功能 |
|------|------|
| `Register()` | 注册一个键和对应的创建函数，可选优先级和帮助文本 |
| `Has()` | 检查键是否已注册 |
| `Create()` | 通过键创建对象实例（未找到返回 nullptr） |
| `Keys()` | 返回所有已注册的键 |
| `HelpMessage()` | 获取注册项的帮助信息 |

### 宏定义系统

用于简化不同场景下的注册声明和定义：

| 宏 | 用途 |
|----|------|
| `C10_DECLARE_TYPED_REGISTRY` | 声明任意键和指针类型的注册表 |
| `C10_DEFINE_TYPED_REGISTRY` | 定义注册表（含警告机制） |
| `C10_REGISTER_TYPED_CREATOR` | 注册自定义创建函数 |
| `C10_REGISTER_TYPED_CLASS` | 注册类（自动生成创建器） |
| `C10_DECLARE_REGISTRY` / `C10_DEFINE_REGISTRY` | 简化版本（键固定为 std::string，指针为 unique_ptr） |
| `C10_DECLARE_SHARED_REGISTRY` / `C10_DEFINE_SHARED_REGISTRY` | shared_ptr 版本 |

### 设计特点

- **静态初始化时注册**：避免全局初始化顺序问题
- **避免 glog 依赖**：使用 fprintf 而非 TORCH_CHECK
- **灵活的优先级机制**：支持默认实现被优先级更高的实现覆盖
- **平台兼容性**：显式处理 Windows DLL 导入导出（dllimport/dllexport）
- **测试友好**：可通过 SetTerminate(false) 让异常替代 std::exit

### 应用场景

- 操作符注册（不同后端的实现）
- 数据类型处理器注册
- 设备类型特定的功能注册
- 插件系统

### 关键特性

- **线程安全**：互斥锁保护注册和查询
- **优先级管理**：高优先级实现可覆盖默认实现
- **动态创建**：通过键动态创建对象，避免硬编码依赖
- **帮助系统**：可为每个注册项关联说明文本
