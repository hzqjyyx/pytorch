这个文件实现了PyTorch类型系统中的**Union类型**和**Optional类型**，用于表示可以持有多种类型值的类型。

## 核心功能

### 1. OptionalType（可选类型）
- 本质是 `Union[T, None]` 的特殊形式
- 提供工厂方法创建包含某个类型和None的联合类型
- `ofTensor()` 返回 `Optional[Tensor]` 的单例

### 2. UnionType（联合类型）
表示可以持有多个不同类型的值，如 `Union[int, str, float]`

**创建过程的标准化**：
- `flattenUnion()` (65-81行)：展开嵌套的Union/Optional，如 `Union[Union[int, str], float]` → `Union[int, str, float]`
- 特殊处理 `NumberType`：展开为 `int, float, complex` 三种类型
- `filterDuplicateSubtypes()` (92-135行)：合并重复和子类型
  - 使用 `unifyTypes` 寻找共同父类型
  - 从右到左遍历，逐步合并可统一的类型
- `sortUnion()` (139-151行)：按kind和字符串表示排序，保证相同Union的一致性

### 3. 类型关系判断

**`equals()`** (295-326行)：
- UnionType vs UnionType：检查所有包含类型是否相同
- UnionType vs OptionalType：尝试转换为Optional再比较
- 特殊处理 `NumberType` 的相等性

**`isSubtypeOfExt()`** (328-360行)：
- 检查Union中每个类型是否都是目标类型的子类型
- 支持与Union、Optional、NumberType的子类型判断
- 提供 `why_not` 诊断信息

### 4. 特殊优化

**Optional自动识别** (232-273行 `create()`方法)：
- 检测到 `Union[T, None]` 自动转换为 `OptionalType`
- 检测 `Union[int, float, complex, None]` → `Optional[Number]`

### 5. 工具方法

- `canHoldType()` (419-430行)：判断Union是否能持有某个类型
- `subtractTypeSet()` (275-277行)：从Union中移除指定类型集合
- `toOptional()` (279-293行)：尝试将Union转换为Optional
- `unionStr()` (362-409行)：生成字符串表示，格式如 `Union[int, str]`

## 设计要点

1. **类型去重与合并**：避免 `Union[int, int, str]` 这种冗余
2. **嵌套展平**：`Optional[Optional[int]]` → `Optional[int]`
3. **排序保证**：使不同创建路径的相同Union可以比较相等
4. **NumberType特殊处理**：作为int/float/complex的抽象
5. **单元素Union禁止**：标准化后只剩一个类型会触发断言（201-216行）

---

**ROCm相关**：无

**Backward相关**：无（此文件只涉及类型系统定义，不涉及自动微分）
