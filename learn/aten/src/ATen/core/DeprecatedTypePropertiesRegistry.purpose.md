## DeprecatedTypePropertiesRegistry 功能分析

这两个文件实现了一个**类型属性注册表**，用于管理和访问已弃用的类型属性。

### 核心设计

**DeprecatedTypePropertiesRegistry** 是一个单例模式的注册表，维护一个二维数组：
- 第一维：后端类型（Backend）- 对应不同的计算后端
- 第二维：标量类型（ScalarType）- 对应不同的数据类型

每个组合都对应一个 **DeprecatedTypeProperties** 实例。

### 关键实现细节

**头文件（.h）**：
- 声明注册表类和删除器
- 使用 `std::unique_ptr` 二维数组存储类型属性
- 数组大小在编译时由 `Backend::NumOptions` 和 `ScalarType::NumOptions` 决定

**实现文件（.cpp）**：
- 构造函数：遍历所有后端-标量类型组合，为每个创建 DeprecatedTypeProperties 实例
- `getDeprecatedTypeProperties()`：通过后端和标量类型查询相应的属性对象
- `globalDeprecatedTypePropertiesRegistry()`：提供全局单例访问点

### 主要功能

- **向后兼容性保证**：保留已弃用的类型属性机制，确保旧代码继续工作
- **集中管理**：所有类型属性通过单一注册表统一管理
- **快速查询**：基于数组索引的 O(1) 时间复杂度查询

### 核心特性列表

• 二维注册表存储结构（后端 × 标量类型）
• 单例模式全局访问
• 编译时确定数组大小
• 每个类型组合对应唯一的属性实例
• 支持后端和标量类型的任意组合
• 用于维护已弃用 API 的向后兼容性
