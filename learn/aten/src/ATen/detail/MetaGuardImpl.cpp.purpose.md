- **文件位置**: `aten/src/ATen/detail/MetaGuardImpl.cpp`

- **主要功能**: 为 Meta 设备类型注册设备保护实现

- **核心内容**:
  - 包含设备保护接口头文件 (`DeviceGuardImplInterface.h`)
  - 使用 `C10_REGISTER_GUARD_IMPL` 宏为 `DeviceType::Meta` 注册一个 `NoOpDeviceGuardImpl` 实现
  - `NoOpDeviceGuardImpl` 是一个空操作实现，不做实际的设备切换操作

- **作用**: 为 Meta 张量后端提供设备上下文管理，虽然 Meta 是虚拟设备但仍需要遵循设备保护的接口约定
