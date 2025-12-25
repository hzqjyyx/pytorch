**文件主要功能：**

- 定义了 `metaFallback` 函数，作为 Meta tensors 的后备实现
- 当操作没有注册 fake impl 或 Meta kernel 时触发该函数
- 抛出 `TORCH_CHECK_NOT_IMPLEMENTED` 异常，提示用户需要添加 fake impl
- 使用 `TORCH_LIBRARY_IMPL` 宏为所有操作（`_`）在 Meta dispatch key 下注册该 fallback
- 异常信息指导用户查看官方教程以支持 PT2 编译 API（torch.compile/torch.export）

**核心目的：** 为未实现 Meta 支持的操作提供有意义的错误提示，而不是静默失败
