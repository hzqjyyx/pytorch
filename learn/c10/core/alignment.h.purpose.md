* **内存对齐常量定义**：定义了全局的内存对齐大小 `gAlignment`
  * 移动设备（C10_MOBILE）：16 字节对齐（适配 ARM NEON 和早期 x86）
  * 其他平台：64 字节对齐（支持 AVX512 等高级 SIMD 指令）

* **分页相关常量**：
  * `gPagesize`：4096 字节（标准内存页大小）
  * `gAlloc_threshold_thp`：2MB（Transparent Huge Pages 的启用阈值）

* **用途**：为 c10 库的内存分配和数据对齐提供跨平台配置
