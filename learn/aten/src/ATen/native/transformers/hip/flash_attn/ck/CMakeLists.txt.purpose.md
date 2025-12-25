- **生成前向传播kernel列表**：通过Python脚本 `generate.py` 调用composable_kernel的FMHA生成器，输出kernel blob列表到 `fwd_blob_list.txt`

- **生成反向传播kernel列表**：同样通过 `generate.py` 生成反向传播的kernel列表到 `bwd_blob_list.txt`

- **编译kernel源文件**：执行 `generate.py` 的前向和反向模式，在当前目录生成实际的kernel源代码文件

- **替换宏名称**：运行 `add_make_kernel_pt.sh` 脚本，将生成的kernel文件中的 `make_kernel` 宏改为 `make_kernel_pt`（PyTorch特定版本）

- **转换文件扩展名**：将所有生成的 `.cpp` 文件重命名为 `.hip` 扩展名（HIP是ROCm的C++方言）
