这个文件定义了 PyTorch 函数化(Functionalization)机制中的逆操作(Inverse)接口。

**主要内容：**

- **枚举 `InverseReturnMode`**：控制函数化逆操作的返回模式- `AlwaysView`：总是返回视图(view)
  - `NeverView`：总是返回副本(copy)
  - `ViewOrScatterInverse`：优先返回视图，除非存在散射(scatter)逆操作则使用后者

- **结构体 `FunctionalInverses`**：包含所有函数化逆操作的声明
  - `${view_inverse_declarations}`：由代码生成器填充的视图逆操作声明
  - `chunk_inverse()`：分块操作的逆操作
  - `narrow_inverse()`：切片操作的逆操作

- **设计特点**：
  - 这是一个模板文件(`.h`)，由代码生成器处理
  - `chunk_inverse` 和 `narrow_inverse` 是手动实现，不由代码生成器生成
  - 用于支持 PyTorch 的函数化变换(Functionalization Transform)

**核心作用**：为可变操作(in-place operations)提供逆向映射，使其能在函数化上下文中正确处理视图和副本的语义。
