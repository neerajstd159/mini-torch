# mini-torch

A tiny PyTorch-inspired tensor implementation
written from scratch in C++.

## Current scope

The current implementation contains:

- Storage
- TensorImpl
- Tensor
- Tensor creation
- CPU memory
- dtype
- device metadata
- sizes
- strides
- storage offset
- contiguous tensors
- tensor indexing at TensorImpl level
- shared storage
- view
- clone
- empty()
- zeros()
- ones()
- full()

## Current limitations

The current creation API uses:

- Float32 as the default dtype
- CPU as the default device

Not implemented yet:

- public Tensor indexing
- tensor iteration
- reshape API
- arbitrary strides
- operators
- broadcasting
- dispatch
- autograd
- CUDA
- Python bindings
- nn
- optimizers
- serialization

## Architecture

```text
                         torch API
                             |
              +--------------+--------------+
              |                             |
              v                             v
       Tensor creation                    Tensor
       empty/zeros/ones/full                |
              |                             |
              +-------------+---------------+
                            |
                            v
                       TensorImpl
                            |
              +-------------+-------------+
              |             |             |
            dtype        sizes/strides   device
                            |
                            v
                         Storage
                            |
                            v
                       raw memory