<!-- # OMEinsum -->
<div align="center"> <img
src="ome-logo.png"
alt="OMEinsum logo" width="510"></img>
<h1>OMEinsum - One More Einsum</h1>
</div>

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://under-Peter.github.io/OMEinsum.jl/dev)
[![CI](https://github.com/under-Peter/OMEinsum.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/under-Peter/OMEinsum.jl/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/under-Peter/OMEinsum.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/under-Peter/OMEinsum.jl)

This is a repository for the _Google Summer of Code_ project on *Differentiable Tensor Networks*.
It implements one function that both computer scientists and physicists love, the *Einstein summation*

<img alt="einsum definition" src="https://github.com/under-Peter/OMEinsum.jl/blob/master/docs/einsum_define.png?raw=true" width=300/>

To find out the details about einsum, please check out my [nextjournal-article](https://nextjournal.com/under-Peter/julia-summer-of-einsum) or the [numpy-manual](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html).

Einstein summation can be implemented in no more than 20 lines of Julia code, the automatic differentiation is also [straightforward](https://giggleliu.github.io/2019/04/02/einsumbp.html). The main effort of this package is improving the [performance](https://github.com/under-Peter/OMEinsum-Benchmarks) utilizing Julia [multiple dispatch on traits](https://white.ucc.asn.au/2018/10/03/Dispatch,-Traits-and-Metaprogramming-Over-Reflection.html). So that people can enjoy the speed of faster specific implementations like BLAS functions, `sum` and `permutedims` on both CPU and GPU without suffering from runtime overhead.

*Note: why the test coverage is not 100%* - GPU-code coverage is not evaluated although we test the GPU code properly on gitlab. Ignoring the GPU-code, the actual coverage is at about _97%_.

*Warning: since v0.4, OMEinsum does not optimize the contraction order anymore. One has to use nested einsum to specify the contraction order manually, e.g. `ein"(ijk,jkl),klm->im"(x, y, z)`.* Please check out the [documentation](https://under-Peter.github.io/OMEinsum.jl/dev/contractionorder/) for more details.

## Install

To install, type `]` in a julia (>=1.5) REPL and then input
```julia pkg
pkg> add OMEinsum
```

## Learn by Examples
To avoid runtime overhead, we recommend users to use [non-standard string literal](https://docs.julialang.org/en/v1/manual/metaprogramming/#Non-Standard-String-Literals-1) `@ein_str`. The following examples illustrates how `einsum` works

```julia
julia> using OMEinsum, SymEngine

julia> catty = fill(Basic(:🐱), 2, 2)
2×2 Array{Basic,2}:
 🐱  🐱
 🐱  🐱

julia> fish = fill(Basic(:🐟), 2, 3, 2)
2×3×2 Array{Basic,3}:
[:, :, 1] =
 🐟  🐟  🐟
 🐟  🐟  🐟

[:, :, 2] =
 🐟  🐟  🐟
 🐟  🐟  🐟

julia> snake = fill(Basic(:🐍), 3, 3)
3×3 Array{Basic,2}:
 🐍  🐍  🐍
 🐍  🐍  🐍
 🐍  🐍  🐍

julia> medicine = ein"ij,jki,kk->k"(catty, fish, snake)
3-element Array{Basic,1}:
 4*🐱*🐍*🐟
 4*🐱*🐍*🐟
 4*🐱*🐍*🐟

julia> ein"ik,kj -> ij"(catty, catty) # multiply two matrices `a` and `b`
2×2 Array{Basic,2}:
 2*🐱^2  2*🐱^2
 2*🐱^2  2*🐱^2

julia> ein"ij -> "(catty)[] # sum a matrix, output 0-dimensional array
4*🐱

julia> ein"->ii"(asarray(snake[1,1]), size_info=Dict('i'=>5)) # get 5 x 5 identity matrix
5×5 Array{Basic,2}:
 🐍  0  0  0  0
 0  🐍  0  0  0
 0  0  🐍  0  0
 0  0  0  🐍  0
 0  0  0  0  🐍
```

Alternatively, people can specify the contraction with a construction approach, which is useful when the contraction code can only be obtained at run time
```julia
julia> einsum(EinCode((('i','k'),('k','j')),('i','j')),(a,b))
```
or a macro based interface, `@ein` macro,
which is closer to the standard way of writing einsum-operations in physics
```julia
julia> @ein c[i,j] := a[i,k] * b[k,j];
```

It is sometimes helpful to specify the order of operations, by inserting brackets, either because you know this will be more efficient,  or to help the computer see what kernels can be used.  For example:
```julia
julia> @ein Z[o,s] := x[i,s] * (W[o,i,j] * y[j,s]);   # macro style

julia> Z = ein"is, (oij, js) -> os"(x, W, y);         # string style
```
This performs matrix multiplication (summing over `j`) 
followed by batched matrix multiplication (summing over `i`, batch label `s`). 
Without the brackets, instead it uses the fallback `loop_einsum`, which is slower.
Calling `allow_loops(false)` will print an error to help you spot such cases:
```julia
julia> @ein Zl[o,s] := x[i,s] * W[o,i,j] * y[j,s];

julia> Z ≈ Zl
true

julia> allow_loops(false);

julia> Zl = ein"is, oij, js -> os"(x, W, y);
┌ Error: using `loop_einsum` to evaluate
│   code = is, oij, js -> os
│   size.(xs) = ((10, 50), (20, 10, 10), (10, 50))
│   size(y) = (20, 50)
└ @ OMEinsum ~/.julia/dev/OMEinsum/src/loop_einsum.jl:26
```

## Comparison with other packages
Similar packages include:
- [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) and [TensorKit.jl](https://github.com/Jutho/TensorKit.jl)
- [ITensors.jl](https://github.com/ITensor/ITensors.jl)

Comparing with the above packages, `OMEinsum` is optimized over large scale tensor network (or einsum, sum-product network) contraction.

## Contribute

Suggestions and Comments in the [_Issues_](https://github.com/under-Peter/OMEinsum.jl/issues) are welcome.