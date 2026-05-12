# ml-lib

A deep learning framework built from scratch in C++/CUDA. It implements a PyTorch-style tensor library with automatic differentiation, GPU execution, neural network modules, and optimizers — with two end-to-end training examples.

## What it is

This is a from-scratch implementation of the core machinery behind modern deep learning:

- **Tensor engine** with CPU and CUDA backends, strided layout, views, and device transfers
- **Reverse-mode autograd** via a dynamic computation graph (topological sort + backward pass)
- **Neural network modules** following a `Module` interface with parameter registration
- **Optimizers**: SGD and Adam (with bias correction)
- **Ops**: each operation ships a forward kernel (`.cpp` / `.cu`) and a corresponding `GradFn`

Two training pipelines are included:

| Example | Task | Architecture | Dataset |
|---|---|---|---|
| `transformer()` | Language modeling | GPT (causal decoder) | Simple English Wikipedia |
| `dcn()` | Binary classification | Deep & Cross Network | UCI Adult Income |

## Architecture

```
src/
├── core/           # Tensor, TensorImpl, Storage, Layout, Shape, Device
├── grad/           # AutogradEngine, GradFn, AutogradMeta
├── ops/            # Differentiable kernels (CPU + CUDA)
│   ├── matmul/
│   ├── softmax/
│   ├── attention/
│   ├── layer_norm/
│   ├── cross_entropy/
│   ├── bce_logits_loss/
│   ├── relu/, add/, mul/, div/, sqrt/
│   ├── gather/, concat/, transpose/
│   └── mask/, repeat/, fill/, rand_fill/, copy/, contiguous/
├── modules/        # Neural network modules
│   ├── Module.h          # Base class
│   ├── Linear, Embedding, DCNEmbedding
│   ├── MultiHeadAttention
│   ├── LayerNorm, MLP
│   ├── TransformerBlock, CrossLayer
│   ├── EncoderBlock, DecoderBlock
│   ├── Transformer, GPT, DCN
├── optim/          # SGD, Adam
└── datasets/       # Loaders for SimpleEnglish, WikiText-103, UCI Adult, MNIST, Iris
```

### Tensor

`Tensor` is a reference-counted wrapper around `TensorImpl`, which holds a `Storage` (the raw allocation, CPU or CUDA) and a `Layout` (shape + strides). This enables zero-copy views, slices, and transposes.

```cpp
Tensor x(Shape({64, 128}), Device::CUDA, /*requires_grad=*/true);
Tensor y = x.transpose(0, 1);   // view, no copy
Tensor z = x.slice(0, 0, 32);   // view into first 32 rows
```

### Autograd

Every differentiable op creates a `GradFn` and stores parent tensors. Calling `.backward()` on a scalar triggers `AutogradEngine::propagate`, which builds the reverse topological order and calls each `GradFn::backward`.

```cpp
Tensor loss = cross_entropy(logits, targets);
loss.backward();   // populates .grad() on all requires_grad tensors
```

### Modules

All modules inherit from `Module` and implement `forward`. Parameters and submodules are registered by name, and `parameters()` recursively collects everything for the optimizer.

```cpp
GPT model(vocab_size, /*d_model=*/128, /*heads=*/4, /*layers=*/2, /*d_ff=*/512, /*ctx=*/64, Device::CUDA);
auto params = model.parameters();
Adam adam(params);
```

### Ops

Each op follows a consistent pattern:

```
ops/matmul/
    matmul.h          # public API
    matmul.cpp        # CPU kernel + AutogradOp wiring
    matmul.cu         # CUDA kernel
    MatmulGradFn.h/.cpp  # backward pass
```

The `AutogradOp` class handles output tensor creation, `requires_grad` propagation, and `GradFn` attachment — so individual kernels stay focused on computation.

## Building

**Requirements:**
- CUDA toolkit with `nvcc`
- GPU with compute capability `sm_89` (Ada Lovelace — RTX 4090/4080). Edit `CFLAGS` in `Makefile` for other architectures.
- C++17

```bash
make        # builds ./a.out
make clean  # removes the binary
```

## Running

Switch the `main.cpp` entry point between the two demos:

```cpp
// Language model training on Simple English Wikipedia
#include "transformer.h"
int main() { transformer(); }

// DCN binary classification on UCI Adult Income
#include "dcn.h"
int main() { dcn(); }
```

Then rebuild and run:

```bash
make && ./a.out
```

### GPT training output (sample)

```
Loading SimpleEnglish dataset...
Dataset loaded. Vocab size: 8143
Total tokens: 1843200
Creating GPT model...
Starting training...
Step 30 | Loss: 4.21 | Avg Loss: 5.03 | Progress: 0.16%
Predicted next token: the
```

### DCN training output (sample)

```
Avg Loss: 0.634
Correct rate: 71
Positive predictions: 18
```

## Datasets

| Dataset | Location | Used for |
|---|---|---|
| Simple English Wikipedia | `datasets/simple_english/` | GPT language modeling |
| WikiText-103 | `datasets/wikitext103/` | Alternative LM corpus |
| UCI Adult Income | `datasets/uci_adult_income/` | DCN binary classification |
| MNIST | `datasets/mnist/` | Available for custom experiments |
| Iris | `datasets/iris/` | Available for custom experiments |

## GPT configuration

| Hyperparameter | Default |
|---|---|
| `d_model` | 128 |
| `num_heads` | 4 |
| `num_layers` | 2 |
| `d_ff` | 512 |
| `context_len` | 64 |
| `learning_rate` | 1e-4 |
| Optimizer | Adam |

## DCN configuration

| Hyperparameter | Value |
|---|---|
| Embedding dim | 8 per category |
| Cross layers | 3 |
| Tower hidden | [64, 32] |
| Learning rate | 1e-4 |
| Optimizer | SGD |
