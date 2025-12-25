## RefcountedDeleter 核心功能分析

### **主要目的**
实现跨多个Python解释器（MultiPy场景）共享Storage数据的机制。当StorageImpl与单个解释器绑定以管理PyObject生命周期时，通过引用计数的DataPtr允许多个StorageImpl实例指向同一块内存。

---

### **关键结构**

**RefcountedDeleterContext** (h:26-32)
- 持有原始的context和deleter函数指针
- 包含原子操作的引用计数器
- 初始refcount为1

---

### **核心函数**

**refcounted_deleter()** (cpp:7-15)
- DataPtr的自定义删除器函数
- 递减引用计数
- 当refcount==0时，删除RefcountedDeleterContext对象本身
- 原始的deletion逻辑由wrapped的deleter负责

**maybeApplyRefcountedDeleter()** (cpp:19-46)
- 检查Storage的DataPtr是否已使用refcounted_deleter
- 如果未使用，则包装现有的context/deleter
- 创建新的RefcountedDeleterContext并替换DataPtr
- 使用互斥锁保护并发访问

**newStorageImplFromRefcountedDataPtr()** (cpp:48-76)
- 创建指向同一数据的新StorageImpl
- 先调用maybeApplyRefcountedDeleter确保DataPtr支持引用计数
- 创建新DataPtr（共享context和deleter）
- **关键**：立即递增refcount（注释说明任何异常都会导致refcount不匹配）
- 返回新的Storage对象

---

### **执行流程**
```
原始Storage (refcount=1)
    ↓
maybeApplyRefcountedDeleter() 包装现有deleter
    ↓
newStorageImplFromRefcountedDataPtr() 创建新Storage
    ↓
refcount++ (现在=2)
    ↓
多个StorageImpl共享同一内存，各自独立生命周期
    ↓
Storage销毁→refcounted_deleter()→refcount--
    ↓
最后一个Storage销毁时refcount=0→删除RefcountedDeleterContext
```

---

### **设计要点**

- **线程安全**：用mutex保护DataPtr替换操作
- **生命周期管理**：ref counting确保内存在所有Storage销毁后才释放
- **关键不变量**：refcount递增必须紧邻新DataPtr创建，防止异常导致计数不准确
- **嵌套删除器**：新deleter包装旧deleter，保持原始deletion逻辑的执行

---

### **Bullet Points**

- 实现跨解释器Storage共享的引用计数机制
- RefcountedDeleterContext封装原始context和deleter
- 自定义refcounted_deleter在引用计数为0时触发真实删除
- maybeApplyRefcountedDeleter延迟包装已有DataPtr
- newStorageImplFromRefcountedDataPtr创建新的共享StorageImpl实例
- 线程安全的DataPtr替换（互斥锁保护）
- 关键设计：refcount递增需紧邻DataPtr创建以防异常
