这个文件是 PyTorch 的 Float8_e4m3fn 浮点类型的内联实现文件。Float8_e4m3fn 是一种 8 位浮点格式，其中 4 位用于指数，3 位用于尾数。

**主要功能包括：**

- **构造函数**：从 float 值初始化 Float8_e4m3fn 对象
- **隐式转换**：Float8_e4m3fn 可以隐式转换回 float 类型
- **特殊值检查**：isnan() 方法用于检测 NaN 值
- **基础算术运算**：支持 +、-、*、/ 运算符，以及对应的复合赋值运算符（+=、-=、*=、/=）
- **混合类型算术**：支持 Float8_e4m3fn 与 float、double、int、int64_t 的混合运算
- **std::numeric_limits 特化**：为标准库提供 Float8_e4m3fn 的数值极限和特性信息，包括：
  - 位数、指数范围等格式参数
  - min()、max()、lowest() 等极值函数
  - epsilon()、denorm_min() 等精度相关信息
  - NaN 和非规范数的支持情况
