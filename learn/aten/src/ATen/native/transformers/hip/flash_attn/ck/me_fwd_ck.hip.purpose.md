**Flash Attention Forward Pass - HIP/CK Backend Implementation**

This file implements the forward pass for memory-efficient attention using the Composable Kernel (CK) backend on HIP (AMD GPUs).

**Main Function: `mem_eff_forward_ck`**

The function wraps the CK implementation and handles two execution paths:

1. **Standard Path** (when `seqstart_q` is not set):
   - Calls `mha_fwd_ck()` with the following parameters:
   - Query (q), Key (k), Value (v) tensors
   - Dropout probability and optional attention bias
   - Causal masking support (is_causal flag)
   - Softmax scaling factor
   - Random number generator for dropout

2. **Nested Tensor Path** (when `seqstart_q` is set):
   - Currently unsupported - raises a `TORCH_CHECK` error
   - Contains commented-out code for variable-length sequences that would call `mha_varlen_fwd_ck()`

**Return Values:**

The function returns 8 tensors as a tuple:
- Output tensor
- Saved Q, K, V (for backward pass)
- Log-sum-exp (LSE) for numerical stability
- Random seed and offset (for dropout reproducibility)
- Dropout random values (optional)

**Key Characteristics:**

- Hard-coded window sizes (both set to -1, meaning full attention)
- Validates that cu_seqlens tensors are either both set or both unset
- Wrapped in `#if defined(USE_CK_FLASH_ATTENTION)` compilation guard
- Part of `pytorch_flash` namespace

**Summary:**
- Delegates to CK backend for standard attention computation
- Supports variable-length sequences in commented-out code (not yet implemented)
- Handles dropout and causal masking through CK kernels
