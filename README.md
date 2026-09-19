# mini-torch

A tiny PyTorch-inspired tensor implementation written from scratch in C++.

## Current scope

The current implementation contains:

- Storage
- TensorImpl
- CPU memory
- dtype
- device metadata
- sizes
- strides
- storage offset
- contiguous tensors
- tensor indexing
- shared storage
- view
- clone

Not implemented yet:

- Tensor
- autograd
- operators
- dispatch
- CUDA
- Python bindings
- nn
- optimizers
- serialization

## Architecture

```text
Tensor
   |
   v
TensorImpl
   |
   +---- dtype
   +---- device
   +---- sizes
   +---- strides
   +---- storage_offset
   |
   v
Storage
   |
   v
raw memory