# OMEinsum.jl

This package provides
- The einsum notation, which is similar to the einsum function in `numpy`, although some details are different.
- Highly optimized algorithms to optimize the contraction of tensors.

The source code is available at [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl).

## Quick start

You can find a set up guide in the [README](https://github.com/under-Peter/OMEinsum.jl). To get started, open a Julia REPL and type the following code.

```@repl intro
using OMEinsum
code = ein"ij,jk,kl,lm->im" # define the einsum operation
optcode = optimize_code(code, uniformsize(code, 100), TreeSA())  # optimize the contraction order
optcode(randn(100, 100), randn(100, 100), randn(100, 100), randn(100, 100))  # compute the result
```