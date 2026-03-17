#include "GatherGradFn.h"
#include "gather.h"
#include <stdexcept>
#include <cmath>

void launch_gather_grad_cuda(
    const int* indices,
    const float* grad_output,
    float* grad_embedding,
    int N,
    int D,
    int V
);

void GatherGradFn::cpu_implementation(const Tensor grad_output) {
    Tensor indices = parents_[0];
    Tensor embedding = parents_[1];
    
    // Only compute gradient for embedding (indices are not differentiable)
    if (embedding.requires_grad()) {
        int N = indices.numel();
        int D = embedding.shape()[1];
        int V = embedding.shape()[0];
        
        const float* indices_data = indices.contiguous().data();
        const float* grad_out = grad_output.data();
        float* grad_emb = embedding.grad().data();
        
        // Scatter gradients: for each index i, add grad_output[i] to grad_embedding[indices[i]]
        for (int i = 0; i < N; ++i) {
            int idx = static_cast<int>(std::round(indices_data[i]));
            
            if (idx >= 0 && idx < V) {
                for (int j = 0; j < D; ++j) {
                    grad_emb[idx * D + j] += grad_out[i * D + j];
                }
            }
        }
    }
}

void GatherGradFn::cuda_implementation(const Tensor grad_output) {
    Tensor indices = parents_[0];
    Tensor embedding = parents_[1];
    
    if (embedding.requires_grad()) {
        int N = indices.numel();
        int D = embedding.shape()[1];
        int V = embedding.shape()[0];
        
        const int* indices_data = reinterpret_cast<const int*>(indices.contiguous().data());
        const float* grad_out = grad_output.data();
        float* grad_emb = embedding.grad().data();
        
        launch_gather_grad_cuda(indices_data, grad_out, grad_emb, N, D, V);
    }
}

