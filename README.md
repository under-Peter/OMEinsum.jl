<!-- # OMEinsum -->
<div align="center"> <img
src="ome-logo.png"
alt="OMEinsum logo" width="510"></img>
<h1>OMEinsum - One More Einsum</h1>
</div>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://under-Peter.github.io/OMEinsum.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://under-Peter.github.io/OMEinsum.jl/dev)
[![Build Status](https://travis-ci.com/under-Peter/OMEinsum.jl.svg?branch=master)](https://travis-ci.com/under-Peter/OMEinsum.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/under-Peter/OMEinsum.jl?svg=true)](https://ci.appveyor.com/project/under-Peter/OMEinsum-jl)
[![Codecov](https://codecov.io/gh/under-Peter/OMEinsum.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/under-Peter/OMEinsum.jl)

This is a repository for the _Google Summer of Code_ project on Differentiable Tensor Networks.
It is a work in progress and will **change substantially this summer (2019)** - no guarantees can be made.

This package exports two functions, `einsum` and `einsumopt`.
`einsum` implements functionality similar to the `einsum` function in `numpy`,
although some details are different.
`einsumopt` receives the same arguments as `einsum` but optimizes the order
of operations that are evaluated internally which might lead to better performance
in some cases.

`einsum` operations are specified by a tuple of tensors `xs = (x1, x2, x3...)`
, a tuple of index-labels for the tensors in `xs`, `ixs = (ix1, ix2, ix3...)`,
and output index-labels `iy` specified as `einsum(ixs, xs, iy)`.

Let `l` be the set of all unique labels in the `ixs` without the ones in `iy`.
`einsum` then calculates an output tensor `y` with indices labelled `iy` according
to the following specification:
```
∀ iy : y[iy] = ∑ₗ x1[ix1] * x2[ix2] * x3[ix3] ...
```
where the sum over `l` implies the sum over all possible values of the labels in `l`.



[Benchmarks are available here](https://github.com/under-Peter/OMEinsum-Benchmarks)

## Examples
Consider multiplying two matrices `a` and `b` which we specify with
```julia
julia> a, b = rand(2,2), rand(2,2);

julia> einsum((('i','k'),('k','j')), (a,b), ('i','j'))
```

`einsum` might also be used in a way closer to the use in `numpy`, via a string specification
such as:
```julia
julia> einsum("ij,jk -> ik", (a,b)) ≈ a*b
true
```

The string parsing introduces a small overhead compared to writing the indices as tuples, but for operations that take on the order of ms this is often negligible.

To find out the details about einsum, check out my [nextjournal-article](https://nextjournal.com/under-Peter/julia-summer-of-einsum) or the [numpy-manual](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html).

If we're interested in the sum of all elements of a matrix product `a*b`
we can reduce over all indices with the specification `ij,jk -> `
```julia
julia> einsum("ij,jk ->", (a,b))[] ≈ sum(a * b)
true
```

Note the use of `[]` to extract the element of a 0-dimensional array.
`einsum` always returns arrays so scalars are wrapped in 0-dimensional arrays.

`einsumopt` will calculate the cost of each possible sequence of operations and evaluate
the (possibly nonunique) optimal operations order.
This is currently associated with a rather large overhead,
but an example from physics with unfortunate default evaluation order shows that in some
cases it might still be worth it:
```julia
julia> d = 5; χ = 50; a = randn(χ,χ); b= randn(χ,d,χ); c = randn(d,d,d,d);

julia> @btime einsum((('x','y'), ('x','k','l'), ('y','m','n'), ('k','m','o','p')), ($a, $b, $b, $c), ('l','n','o','p'));
  1.323 s (473 allocations: 2.33 GiB)

julia> @btime einsumopt((('x','y'), ('x','k','l'), ('y','m','n'), ('k','m','o','p')), ($a, $b, $b, $c), ('l','n','o','p'));
  2.845 ms (11733 allocations: 2.00 MiB)

```
although the same effect can be had by choosing the labels appropriately, such that
the best contraction sequence is in alphabetical order:
```julia
julia> @btime einsum((('i','j'), ('i','k','l'), ('j','m','n'), ('k','m','o','p')), ($a, $b, $b, $c), ('l','n','o','p'));
  645.273 μs (331 allocations: 1.54 MiB)
```
or with a string-specification
```julia
julia> @btime einsum("ij,ikl,jmn,kmop -> lnop", ($a, $b, $b, $c))
  681.774 μs (370 allocations: 1.54 MiB)
```


## Contribute

Suggestions and Comments in the _Issues_ are welcome.

## License
MIT License
