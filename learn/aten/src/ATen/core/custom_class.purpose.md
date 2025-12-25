## 文件功能分析

### custom_class.h
定义了自定义类的类型管理接口：
- `ClassType` 结构体和 `ClassTypePtr` 智能指针类型
- `getCustomClassTypeImpl()` 函数声明，用于通过 `std::type_index` 查找类型
- `getCustomClassType<T>()` 模板函数，提供缓存的类型查询（避免重复哈希查找）

### custom_class.cpp
实现了自定义类的注册、查询和管理系统：

**核心数据结构：**
- `getCustomClassTypeMap()`：全局哈希表，存储 `type_index` → `ClassTypePtr` 的映射
- `customClasses()`：全局哈希表，存储类名字符串 → `ClassTypePtr` 的映射
- `customClassMethods()`：存储自定义类方法的向量

**主要函数功能：**
- `getCustomClassTypeImpl()`：从类型索引查找对应的 `ClassType`。若直接查询失败，通过遍历比较类名处理跨动态库的类型索引问题
- `registerCustomClass()`：将新的自定义类注册到全局映射，防止重复注册
- `getCustomClass()`：按名字查询已注册的自定义类，并记录访问日志（若启用）
- `getAllCustomClassesNames()`：返回所有已注册类的名字集合
- `isCustomClass()`：检查 `IValue` 对象是否为自定义类实例
- `registerCustomClassMethod()`：向类添加新方法
- `customClassSchemasForBCCheck()`：获取所有自定义类方法的函数签名（用于向后兼容性检查）
- `class_base::class_base()`：构造函数，创建新的 `ClassType` 并注册到两个哈希表
- `class_base::withNewArguments()`：创建具有默认参数的新函数签名

**关键特性：**
- 跨共享库的类型识别（通过类名比对而非 `type_index` 直接比对）
- 支持自定义类的属性和方法管理
- 集成 TorchScript JIT 编译系统
- 内置性能记录（RECORD_FUNCTION 机制）

---

### 核心功能总结
- 全局注册表管理 PyTorch 自定义类的元数据
- 支持按类型索引或类名查询类定义
- 处理跨动态库环境下的类型识别问题
- 集成 TorchScript 方法和签名管理
