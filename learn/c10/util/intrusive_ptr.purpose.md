# intrusive_ptr 智能指针实现

## 核心设计理念

PyTorch 的 `intrusive_ptr` 是一个性能优化的智能指针，相比 `std::shared_ptr` 采用**侵入式引用计数**（refcount 存储在对象内部），避免了额外的控制块分配。

## 主要组件

### 1. intrusive_ptr_target（基类）

对象必须继承此类才能使用 `intrusive_ptr` 管理。包含两个原子计数器：

```cpp
mutable std::atomic<uint32_t> refcount_;   // 强引用计数
mutable std::atomic<uint32_t> weakcount_;  // 弱引用计数
```

**引用计数规则**：
- `refcount` = 强引用数量
- `weakcount` = 弱引用数量 + (refcount > 0 ? 1 : 0)
- 不变式：`refcount > 0 => weakcount > 0`

**生命周期管理**：
- `refcount` 归零时：调用 `release_resources()`（虚函数，可重写释放资源），然后 `weakcount--`
- `weakcount` 归零时：`delete` 对象本身

### 2. intrusive_ptr<T>（强引用智能指针）

#### 关键特性

**栈对象安全检测**：
- 构造时初始化 `refcount = 0, weakcount = 0`
- 动态分配对象（如 `make_intrusive`）会将计数设为 1
- 从栈对象创建智能指针会触发断言失败（检测 refcount 是否为 0）

**构造函数设计**：
```cpp
// 公共：默认/nullptr 构造
intrusive_ptr() noexcept;

// 私有：从裸指针构造（仅供 make_intrusive/reclaim 使用）
explicit intrusive_ptr(TTarget* target);

// 公共：标记不增加引用计数的构造
explicit intrusive_ptr(TTarget* target, raw::DontIncreaseRefcount);
```

**所有权转移 API**：
- `release()`: 返回裸指针但不减引用计数，调用者需用 `reclaim()` 重新包装
- `reclaim(ptr)`: 从 `release()` 返回的指针创建智能指针，不增加引用计数
- `reclaim_copy(ptr)`: 创建新引用并增加引用计数

#### 特殊工厂方法

```cpp
// 标准分配方式
make_intrusive<T>(args...)

// 从已分配对象创建（pybind11 使用）
unsafe_steal_from_new(raw_ptr)

// 非堆分配对象适配（静态运行时使用）
unsafe_adapt_non_heap_allocated(raw_ptr, expected_decrefs)
// 设置 refcount = kImpracticallyHugeReferenceCount + expected_decrefs
// 防止对象被释放

// 从非所有权裸指针创建（类似 enable_shared_from_this）
unsafe_reclaim_from_nonowning(raw_ptr)
```

### 3. weak_intrusive_ptr<T>（弱引用智能指针）

**核心操作**：
```cpp
// 升级为强引用（线程安全）
intrusive_ptr<T> lock() const noexcept {
    auto refcount = target_->refcount_.load(seq_cst);
    do {
        if (refcount == 0) return nullptr;  // 对象已销毁
    } while (!compare_exchange_weak(refcount, refcount + 1));
    return intrusive_ptr<T>(target_, DontIncreaseRefcount{});
}

// 检查对象是否已销毁
bool expired() const { return use_count() == 0; }
```

### 4. 内存序要求

```cpp
// 增减操作使用 acq_rel（保证 use_count()/unique() 可靠性）
atomic_refcount_increment: memory_order_acq_rel
atomic_refcount_decrement: memory_order_acq_rel
atomic_weakcount_decrement: memory_order_acq_rel

// 弱引用计数增加使用 relaxed（仅测试用）
atomic_weakcount_increment: memory_order_relaxed

// lock() 的 CAS 使用 seq_cst（防止 ABA 问题）
```

## 实现细节

### reset_() 逻辑

```cpp
void reset_() noexcept {
    if (target_ != NullType::singleton() && 
        atomic_refcount_decrement(target_->refcount_) == 0) {
        
        // 优化：如果 weakcount == 1（只有强引用带来的 +1），直接删除
        bool should_delete = (target_->weakcount_.load(acquire) == 1);
        
        if (!should_delete) {
            target_->release_resources();  // 调用虚函数释放资源
            should_delete = (atomic_weakcount_decrement(weakcount_) == 0);
        }
        
        if (should_delete) {
            delete target_;
        }
    }
}
```

### NullType 模板参数

允许自定义空指针单例：
```cpp
// 默认实现
struct intrusive_target_default_null_type<T> {
    static constexpr T* singleton() { return nullptr; }
};
```

## 辅助功能

**MaybeOwnedTraits 特化**：
支持创建不增加引用计数的"借用"引用：
```cpp
createBorrow(owned) → reclaim(owned.get())
destroyBorrow(borrow) → borrow.release()
```

**原始指针操作命名空间**：
```cpp
raw::intrusive_ptr::incref(ptr)
raw::intrusive_ptr::decref(ptr)
raw::intrusive_ptr::make_weak(ptr)  // 强指针转弱指针

raw::weak_intrusive_ptr::incref(ptr)
raw::weak_intrusive_ptr::decref(ptr)
raw::weak_intrusive_ptr::lock(ptr)   // 弱指针升级
```

**STL 支持**：
- 比较运算符（`<`, `==`, `!=`）支持 `std::map/set`
- `std::hash` 特化支持 `std::unordered_map/set`

---

**其他细节**：
- 拷贝/移动构造时不拷贝引用计数（计数是内存位置的固有属性）
- 析构函数禁用 `-Wterminate` 以允许断言检查
- pybind11 通过友元访问私有构造函数实现自定义 holder
- `.cpp` 文件仅包含头文件（实现均在头文件中）
