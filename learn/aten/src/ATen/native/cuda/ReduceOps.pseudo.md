--------------------------------
归约操作注册
来源: aten/src/ATen/native/cuda/ReduceOps.cpp
--------------------------------

【函数】
    norm_kernel_cuda(iter: TensorIterator, val: Scalar)
        用途: 计算Lp范数归约的CUDA实现
        算法:
            p = convert_to_double(val)
            
            if iter.numel() == 0:
                iter.output().fill_(INFINITY if p < 0 else 0)
                return
            
            norm_launch_kernel(iter, p)
            
            if iter.output().dtype is complex:
                iter.output().imag().zero_()

    min_kernel_impl(result: Tensor, indice: Tensor[int64], self: Tensor, dim: int64, keepdim: bool)
        用途: 沿指定维度求最小值及索引
        算法:
            iter = make_reduction(self, result, indice, dim, keepdim, dtype=self.dtype, index_dtype=int64)
            min_launch_kernel(iter)

    max_kernel_impl(result: Tensor, indice: Tensor[int64], self: Tensor, dim: int64, keepdim: bool)
        用途: 沿指定维度求最大值及索引
        算法:
            iter = make_reduction(self, result, indice, dim, keepdim, dtype=self.dtype, index_dtype=int64)
            max_launch_kernel(iter)

    aminmax_kernel_impl(self: Tensor, dim: int64, keepdim: bool, min_result: Tensor, max_result: Tensor)
        用途: 同时计算最小值和最大值
        算法:
            iter = make_reduction("aminmax_cuda", min_result, max_result, self, dim, keepdim, dtype=self.dtype)
            
            if iter.numel() != 0:
                aminmax_launch_kernel(iter)

    min_all_kernel_impl(result: Tensor, input: Tensor)
        用途: 全张量最小值
        算法:
            iter = make_reduction("min_all", result, input, dims=[], keepdim=false, dtype=input.dtype)
            min_all_launch_kernel(iter)

    max_all_kernel_impl(result: Tensor, input: Tensor)
        用途: 全张量最大值
        算法:
            iter = make_reduction("max_all", result, input, dims=[], keepdim=false, dtype=input.dtype)
            max_all_launch_kernel(iter)

    aminmax_allreduce_kernel_impl(input: Tensor, min_result: Tensor, max_result: Tensor)
        用途: 全张量同时求最小值和最大值
        算法:
            iter = make_reduction("aminmax_cuda", min_result, max_result, input, dims=[], keepdim=false, dtype=input.dtype)
            
            assert iter.numel() > 0, "min_max on a tensor with no elements is not defined."
            
            aminmax_allreduce_launch_kernel(iter)

【注册】
    REGISTER_CUDA_DISPATCH(min_stub, &min_kernel_impl)
    REGISTER_CUDA_DISPATCH(max_stub, &max_kernel_impl)
    REGISTER_CUDA_DISPATCH(min_all_stub, &min_all_kernel_impl)
    REGISTER_CUDA_DISPATCH(max_all_stub, &max_all_kernel_impl)
    REGISTER_CUDA_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_kernel_impl)
    REGISTER_CUDA_DISPATCH(aminmax_stub, &aminmax_kernel_impl)
    REGISTER_CUDA_DISPATCH(norm_stub, &norm_kernel_cuda)

【依赖】
    norm_launch_kernel           # 实际的范数计算kernel (ReduceOps.h)
    min_launch_kernel            # 最小值归约kernel
    max_launch_kernel            # 最大值归约kernel
    aminmax_launch_kernel        # 最小最大值同时归约kernel
    min_all_launch_kernel        # 全张量最小值kernel
    max_all_launch_kernel        # 全张量最大值kernel
    aminmax_allreduce_launch_kernel  # 全张量最小最大值kernel
    meta::make_reduction         # 创建归约迭代器

【备注】
    - 这是CUDA归约操作的分发层，实际kernel实现在.cu文件中
    - 所有kernel通过TensorIterator抽象访问数据
    - norm_kernel_cuda对复数输出会清零虚部
