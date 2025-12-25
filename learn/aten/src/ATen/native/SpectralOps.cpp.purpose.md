# SpectralOps.cpp 主要功能

这个文件实现了 PyTorch 的**频域信号处理操作**，主要是各种**傅里叶变换(FFT)**及其逆变换。

## 核心功能模块

### 1. 类型提升机制 (Lines 74-112)

- `promote_type_fft()`: 将输入类型提升到FFT所需类型
  - 整数 → 默认浮点类型
  - 实数 → 复数（当需要时）
  - 仅CUDA支持半精度(Half)

- `promote_tensor_fft()`: 对张量应用类型提升

### 2. 归一化模式处理 (Lines 117-131)

`norm_from_string()` 将NumPy风格的归一化字符串转换为内部枚举:
- `"backward"`: 前向不归一化，逆向除以n
- `"forward"`: 前向除以n，逆向不归一化  
- `"ortho"`: 双向都除以√n

### 3. 输入调整 (Lines 135-158)

`resize_fft_input()`: 调整输入张量尺寸以匹配FFT要求
- 过小则零填充
- 过大则切片

### 4. 一维FFT实现

#### 4.1 复数到实数 (C2R, Lines 199-220)
`fft_c2r()` - 用于 `irfft`, `hfft`
- 输入: 复数张量（Hermitian对称）
- 输出: 实数张量
- 推断输出长度: `n = 2*(input.size - 1)`

#### 4.2 实数到复数 (R2C, Lines 223-255)
`fft_r2c()` - 用于 `rfft`, `ihfft`
- 输入: 实数张量
- 输出: 复数张量
- 支持 `onesided=True` 利用共轭对称性节省空间

#### 4.3 复数到复数 (C2C, Lines 258-273)
`fft_c2c()` - 用于 `fft`, `ifft`
- 标准复数FFT/IFFT

### 5. 多维FFT实现

#### 5.1 参数规范化 (Lines 284-343)
`canonicalize_fft_shape_and_dim_args()`:
- 处理可选的 `s` (shape) 和 `dim` 参数
- 默认行为: 未指定则使用所有维度
- 检查维度唯一性

#### 5.2 N维复数FFT (Lines 346-355)
`fftn_c2c()` - 用于 `fftn`, `ifftn`, `fft2`, `ifft2`
- 在多个维度上执行C2C变换

#### 5.3 N维实数FFT (Lines 476-502)
`fft_rfftn_impl()` - 用于 `rfftn`, `rfft2`
- 实数输入的多维FFT
- 最后一维利用共轭对称性

#### 5.4 N维Hermitian FFT (Lines 557-643)
- `fft_hfftn_impl()`: Hermitian输入 → 实数输出
- `fft_ihfftn_impl()`: 实数输入 → Hermitian输出
- 需要特殊处理最后一维的对称性

### 6. 频率辅助函数 (Lines 707-749)

- `fft_fftfreq()`: 生成FFT频率坐标
  - 返回 `[0, 1, ..., n/2-1, -n/2, ..., -1] / (n*d)`
  
- `fft_rfftfreq()`: 生成单边FFT频率坐标
  - 返回 `[0, 1, ..., n/2] / (n*d)`

### 7. 频谱移位 (Lines 768-790)

- `fft_fftshift()`: 将零频移到中心 (循环移位 n/2)
- `fft_ifftshift()`: 逆移位 (循环移位 (n+1)/2)

### 8. STFT - 短时傅里叶变换 (Lines 826-1003)

**核心流程:**

1. **参数默认化**
   - `hop_length` 默认 `n_fft/4`
   - `win_length` 默认 `n_fft`

2. **中心化填充** (可选, `center=True`)
   - 在信号两端填充 `n_fft/2` 个零

3. **分帧** (Lines 960-963)
   - 使用 `as_strided` 创建滑窗视图
   - 每帧长度 `n_fft`, 步长 `hop_length`
   - 帧数: `n_frames = 1 + (len - n_fft) / hop_length`

4. **加窗** (Lines 936-947, 964-966)
   - 将窗函数应用到每帧
   - 如果 `win_length < n_fft`, 对窗进行零填充

5. **FFT变换**
   - 复数输入: C2C FFT
   - 实数输入: R2C FFT (可选单边输出)

6. **转置输出**: `(batch, n_frames, fft_size)` → `(batch, fft_size, n_frames)`

**特殊参数:**
- `align_to_window`: 对齐窗口而非FFT长度
- `normalized`: 使用 `by_root_n` 归一化
- `return_complex`: 返回复数或实数视图

### 9. ISTFT - 逆短时傅里叶变换 (Lines 1026-1209)

**核心流程:**

1. **逆FFT** (Lines 1141-1151)
   - 复数: C2C IFFT (归一化 `by_n`)
   - 实数: C2R IFFT

2. **加窗** (Lines 1126-1154)
   - 对IFFT结果乘以窗函数
   - 窗函数需与STFT一致

3. **重叠相加** (Lines 1156-1168)
   - 使用 `unfold_backward` 实现
   - 累加所有重叠帧

4. **归一化** (Lines 1162-1194)
   - 计算窗函数能量包络 `window_envelop`
   - 除以包络以补偿重叠

5. **裁剪** (Lines 1174-1186)
   - 去除中心化填充的零
   - 可选指定输出长度

**关键检查:**
- 窗包络最小值 > 1e-11 (避免除零)
- 输入必须是复数张量

### 10. cuFFT计划缓存管理 (Lines 795-809)

通过CUDA钩子访问:
- `_cufft_get_plan_cache_max_size()`
- `_cufft_set_plan_cache_max_size()`
- `_cufft_get_plan_cache_size()`
- `_cufft_clear_plan_cache()`

### 11. Hermitian对称填充 (Lines 1211-1300)

`_fft_fill_with_conjugate_symmetry_()`:
- 对半Hermitian张量进行共轭镜像填充
- 使用负步长实现内存高效的镜像
- 按步长重排维度优化数据局部性
- 分发到CPU/CUDA特定实现

## 公开API总结

**1D变换:**
- `fft`, `ifft` - 通用复数FFT
- `rfft`, `irfft` - 实数FFT (利用共轭对称)
- `hfft`, `ihfft` - Hermitian FFT (实/复互转)

**2D/ND变换:**
- `fft2`, `ifft2`, `fftn`, `ifftn` - 多维复数FFT
- `rfft2`, `irfft2`, `rfftn`, `irfftn` - 多维实数FFT
- `hfft2`, `ihfft2`, `hfftn`, `ihfftn` - 多维Hermitian FFT

**辅助函数:**
- `fftfreq`, `rfftfreq` - 频率坐标
- `fftshift`, `ifftshift` - 频谱移位
- `stft`, `istft` - 短时傅里叶变换

---

## 其他内容

**ROCm相关:**
- 无直接ROCm代码，通过统一CUDA接口间接支持

**Backward相关:**
- 无显式反向传播实现
- 依赖PyTorch自动微分框架
- FFT操作通过autograd自动计算梯度
