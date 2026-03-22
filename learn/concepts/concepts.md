# PyTorch Core Concepts 文档

## 1. 概述

### 问题背景

当我们开始系统阅读 PyTorch 的 `c10 + aten` 代码时，最先遇到的困难通常不是某一个类太复杂，而是概念太多而且分散。前面看到了 [`c10::TensorImpl`](../../c10/core/TensorImpl.h)、[`c10::DispatchKeySet`](../../c10/core/DispatchKeySet.h)、[`c10::Device`](../../c10/core/Device.h)、[`c10::Stream`](../../c10/core/Stream.h)、[`c10::DeviceGuard`](../../c10/core/DeviceGuard.h)，再往上又会遇到 [`at::TensorBase`](../../aten/src/ATen/core/TensorBase.h)、[`c10::Storage`](../../c10/core/Storage.h)、[`c10::Allocator`](../../c10/core/Allocator.h)、[`c10::Event`](../../c10/core/Event.h)、[`c10::TensorOptions`](../../c10/core/TensorOptions.h)，以及负责算子注册与分发的 [`c10::Dispatcher`](../../aten/src/ATen/core/dispatch/Dispatcher.h)。如果只是把它们当作零散的名词去背，很快就会失去方向，因为你很难判断这些对象到底属于同一层，还是分属不同的职责边界。

真正有效的方式，不是先把每个类逐个抠细节，而是先建立一个整体架构图，把这些概念放回同一张图里。这样做的好处是，你会先知道“哪些对象是在描述 tensor 本身，哪些对象是在描述设备执行环境，哪些对象是在描述算子分发，哪些对象只是后端或上层系统借用的运行时基础设施”。一旦这个大图建立起来，后续再追 `torch.add`、`tensor.to`、`contiguous`、`copy_`、`backward` 这类具体链路时，就不会再把张量对象、执行上下文、内存管理和算子分发混成一团。

### 设计思路

这份文档的目标不是写成完整词典，而是先给出一张适合阅读源码的 `c10 + aten` 概念总图。图中会把核心对象分成几组。最底层是运行时公共抽象，比如 `Device`、`Stream`、`Event`、`Allocator`、`Storage`；中间层是张量对象本体，也就是 `TensorImpl`、`TensorBase` 和 `Tensor`；另一条横向主线是算子系统，包括 `DispatchKeySet`、`Dispatcher`、`native_functions.yaml` 与 backend kernels；最上面再点到 `autograd`、Python bindings 和更高层 frontend，让整张图能说明“边界到哪儿为止”。图里我尽量只保留真正稳定的骨架，不把太多历史细节塞进去，这样它更适合作为后续阅读时反复回看的参照。

## 2. ASCII 架构图

下面这张图可以当成 `c10 + aten` 的整体地图。图中的箭头主要表示依赖关系和阅读时更自然的理解顺序，而不是某一条唯一的运行时调用方向。

```text
                               ┌──────────────────────┐
                               │     Python torch     │
                               │   torch / torch._C   │
                               └──────────┬───────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │     torch/csrc       │
                               │ bindings / autograd  │
                               └──────────┬───────────┘
                                          │
                                          ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                ATen API                                │
    │                                                                        │
    │  ┌────────────────────┐ Extends ┌────────────────────┐                 │
    │  │    at::Tensor      │ ─────►  │   at::TensorBase   │                 │
    │  │ user-facing handle │         │ lightweight handle │                 │
    │  └────────────────────┘         └────────┬───────────┘                 │
    └──────────────────────────────────────────┼─────────────────────────────┘
                                               │
                                               ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                              Tensor Object                             │
    │                                                                        │
    │  ┌───────────────────────────────────────────────────────────────┐     │
    │  │                     c10::TensorImpl                           │     │
    │  │                                                               │     │
    │  │  sizes/strides  storage_offset  dtype  device  version        │     │
    │  │  autograd_meta  key_set         layout memory-format flags    │     │
    │  └───────────┬───────────────────────────┬───────────────────────┘     │
    └──────────────┼───────────────────────────┼─────────────────────────────┘
                   │                           │
                   ▼                           ▼
      ┌───────────────────────────┐   ┌───────────────────────────────┐
      │       c10::Storage        │   │   c10::DispatchKeySet         │
      │ owns memory buffer handle │   │ tensor runtime type tags      │
      └─────────────┬─────────────┘   └───────────────┬───────────────┘
                    │                                 │
                    ▼                                 ▼
      ┌───────────────────────────┐      ┌─────────────────────────────┐
      │      c10::Allocator       │      │      c10::Dispatcher        │
      │ allocate / free / clone   │      │ schema + kernel selection   │
      └─────────────┬─────────────┘      └───────────────┬─────────────┘
                    │                                    │
                    ▼                                    ▼
      ┌───────────────────────────┐      ┌─────────────────────────────┐
      │       c10::DataPtr        │      │ native_functions.yaml       │
      │ ptr + deleter + device    │      │ torchgen / operator glue    │
      └───────────────────────────┘      └───────────────┬─────────────┘
                                                         │
                                                         ▼
                                         ┌───────────────────────────────┐
                                         │ backend kernels / native impl │
                                         │ cpu / cuda / mps / xpu / meta │
                                         └───────────────────────────────┘


    ┌────────────────────────────────────────────────────────────────────────┐
    │                           Execution Context                            │
    │                                                                        │
    │   c10::Device      where to run                                        │
    │   c10::Stream      ordered async execution lane on a device            │
    │   c10::Event       cross-stream synchronization point                  │
    │   c10::DeviceGuard RAII current-device switch                          │
    │   c10::StreamGuard RAII current-stream switch                          │
    └────────────────────────────────────────────────────────────────────────┘


    ┌────────────────────────────────────────────────────────────────────────┐
    │                           Tensor Attributes                            │
    │                                                                        │
    │   c10::ScalarType   dtype                                              │
    │   c10::Layout       strided / sparse / ...                             │
    │   c10::MemoryFormat contiguous / channels_last / ...                   │
    │   c10::TensorOptions factory-time option bundle                        │
    │   c10::Scalar       scalar value abstraction                           │
    │   c10::SymInt       symbolic size / shape integer                      │
    │   c10::Generator    random-number generator state                      │
    └────────────────────────────────────────────────────────────────────────┘
```

## 3. 读图说明

### 张量主线

读这张图时，最重要的一条竖线是 `Tensor -> TensorBase -> TensorImpl`。[`at::Tensor`](../../aten/src/ATen/templates/TensorBody.h) 是用户和大部分 ATen 代码最常接触的张量句柄，[`at::TensorBase`](../../aten/src/ATen/core/TensorBase.h) 是更轻量的中间层，而真正保存张量状态的是 [`c10::TensorImpl`](../../c10/core/TensorImpl.h)。如果你想理解“一个 tensor 到底是什么”，最终一定要落回 `TensorImpl`。它内部既持有形状、步长、dtype、device、storage offset，也持有 version counter、autograd metadata 和 dispatch key set。这就是为什么 `TensorImpl` 是阅读 `c10 + aten` 时最值得先吃透的类。

### 内存主线

另一条重要主线是 `Allocator -> DataPtr -> Storage -> TensorImpl`。[`c10::Allocator`](../../c10/core/Allocator.h) 负责真正分配和释放内存，[`c10::DataPtr`](../../c10/core/Allocator.h) 则把原始指针、deleter 和 device 绑在一起，[`c10::Storage`](../../c10/core/Storage.h) 负责把这块内存包装成可以被张量复用、共享和引用计数管理的对象，最后 [`c10::TensorImpl`](../../c10/core/TensorImpl.h) 再引用 `Storage` 并解释这块内存该如何被看成一个多维 tensor。很多初学者会把 `TensorImpl` 和 `Storage` 混在一起看，但它们的职责其实非常清楚：`Storage` 管“这一块内存”，`TensorImpl` 管“如何把它解释成一个 tensor”。

### 执行主线

`Device`、`Stream`、`Event`、`Guard` 这一组对象描述的是执行环境，而不是 tensor 的数据结构。[`c10::Device`](../../c10/core/Device.h) 回答的是“在哪块设备上”，[`c10::Stream`](../../c10/core/Stream.h) 回答的是“在这块设备上的哪条异步执行序列上”，[`c10::Event`](../../c10/core/Event.h) 回答的是“不同 stream 之间如何同步”，而 [`c10::DeviceGuard`](../../c10/core/DeviceGuard.h) 与 [`c10::StreamGuard`](../../c10/core/StreamGuard.h) 则是当前线程上下文切换的 RAII 工具。阅读 kernel 或后端代码时，如果你看到这些类，应该立刻想到自己已经进入“执行上下文”问题，而不再是单纯的 tensor 元数据问题。

### 算子主线

图中右侧的 `DispatchKeySet -> Dispatcher -> native_functions.yaml -> backend kernels` 是算子系统的主线。[`c10::DispatchKeySet`](../../c10/core/DispatchKeySet.h) 描述一个 tensor 在运行时带有哪些标签，比如 CPU、CUDA、Sparse、Autograd、Meta 等；[`c10::Dispatcher`](../../aten/src/ATen/core/dispatch/Dispatcher.h) 根据 operator schema 和这些 key 选择实际 kernel；[`native_functions.yaml`](../../aten/src/ATen/native/native_functions.yaml) 与 `torchgen` 则负责从算子声明生成 API、schema 注册和 dispatch glue；最后才落到 `native` 或后端目录里的真正实现。把这一条链吃透之后，你再去读 `add`、`sum`、`matmul`、`copy_` 这类算子，就不会只看见一堆分散的注册宏和模板代码，而会知道它们分别处在哪一层。

### 边界感

最上面的 `torch/csrc` 和 Python `torch` 我在图里只做了定位，没有展开，是因为这份文档的重心仍然是 `c10 + aten`。对学习者来说，建立边界感比一开始把所有上层系统都拖进来更重要。你可以先把这张图记成一句话：`c10` 提供通用运行时抽象，`aten` 用这些抽象组织张量和算子，而更高层的 Python、autograd、frontend 系统则建立在这两层之上。只要这个边界足够清楚，后续无论你进入 autograd engine、custom op、还是 backend 实现，都会更容易判断自己当前处在哪一层。
