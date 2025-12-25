# ATen Stack 核心功能

这个文件定义了 PyTorch JIT 执行引擎中的栈操作接口。

## Operation 类
- 包装函数对象，使其能处理 `Stack&` 参数
- 支持两种构造方式：
  - 旧风格：`void(Stack*)` (已弃用)
  - 新风格：`void(Stack&)` (推荐)
- 可通过 `target()` 方法获取底层函数指针

## Stack 定义
- 类型别名：`std::vector<IValue>`
- IValue 是通用值类型，可存储任意 PyTorch 数据

## 栈操作函数

### 读取操作
- `peek(stack, i, N)` - 获取后 N 个元素中的第 i 个（不移除）
- `peekSlice(stack, i, len, N)` - 获取一个切片
- `last(stack, N)` - 获取栈顶 N 个元素

### 弹出操作
- `pop(stack)` - 弹出单个元素
- `pop(stack, n)` - 弹出 n 个元素并返回向量
- `pop(stack, a, b, ...)` - 变参弹出，自动类型转换（`a = pop().to<Type>()`）

### 入栈操作
- `push_one(stack, arg)` - 推入单个元素
- `push(stack, ...)` - 推入多个元素（变参）
- 特殊处理：`TensorOptions` 会被展开成 4 个独立元素

### 清理与打包
- `drop(stack, n)` - 删除后 n 个元素
- `pack(stack, value)` - 推入单个值
- `TuplePacker` - 递归展开并推入元组的所有元素
- `push_list_elements(stack, list)` - 推入列表的所有元素

## 设计原则

**所有函数同时提供两个版本**：接收 `Stack&` 和 `Stack*`

**栈帧约定**：
```
操作前: <其他项> I0, I1, ... IN ← stack.back()
操作后: <其他项> O0, O1, ... OM
```
操作 pop 掉最后 N 个输入，push N 个输出

**关键特性**：支持增量释放张量所有权，减少 GPU 内存占用
