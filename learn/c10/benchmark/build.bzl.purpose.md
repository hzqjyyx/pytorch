这个文件定义了 C10 库的基准测试构建目标。主要功能：

- **定义构建规则函数**：`define_targets(rules)` 接收构建规则对象作为参数
- **配置基准测试二进制文件**：创建名为 `intrusive_ptr` 的可执行文件
- **指定源文件**：使用 `intrusive_ptr_benchmark.cpp` 作为源代码
- **设置构建标签**：标记为 `benchmark` 便于识别和筛选
- **声明依赖项**：
  - `//c10/util:base` - C10 工具库基础模块
  - `@google_benchmark//:benchmark` - Google Benchmark 性能测试框架
- **Bazel 构建系统**：采用 Bazel 作为构建工具（.bzl 为 Bazel 配置文件）
