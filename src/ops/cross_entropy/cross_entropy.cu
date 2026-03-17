__global__ void cross_entropy_forward_kernel(
    const float* logits, 
    const float* targets, 
    float* losses, 
    int batch_size, 
    int num_classes) 
{
    int b = blockIdx.x; // Each block handles one row (one batch element)
    if (b >= batch_size) return;

    const float* row = logits + (b * num_classes);
    int target_idx = static_cast<int>(targets[b]);

    // 1. Find max logit (for numerical stability)
    float max_val = -1e20f;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        max_val = fmaxf(max_val, row[c]);
    }
    // Block-level reduction for max_val would go here for very large num_classes
    // For simplicity, we'll assume num_classes fits reasonably in shared memory or use a loop

    // 2. Compute Log-Sum-Exp
    float sum_exp = 0.0f;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        sum_exp += expf(row[c] - max_val);
    }
    // Finalize sum_exp and max_val across threads (simplified version)
    // ... (Use atomicAdd or __shfl_down_sync for a production-grade reduction)

    if (threadIdx.x == 0) {
        float log_sum_exp = max_val + logf(sum_exp);
        losses[b] = log_sum_exp - row[target_idx];
    }
}