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
and a tuple of index-labels for the tensors in `xs`, `ixs = (ix1, ix2, ix3...)`,
optionally an output index-labels can be specified `iy` as `einsum(ixs, xs, iy)`.

Let `l` be the set of all unique labels in the `ixs` without the ones in `iy`.
`einsum` then calculates an output tensor `y` with indices labelled `iy` according
to the following specification:
```
∀ iy : y[iy] = ∑ₗ x1[ix1] * x2[ix2] * x3[ix3] ...
```
where the sum over `l` implies the sum over all possible values of the labels in `l`.

As an example, consider multiplying two matrices `a` and `b` which we specify with
```julia
julia> a, b = rand(2,2), rand(2,2);

julia> einsum((('i','k'),('k','j')), (a,b), ('i','j'))
```
where we have
* `x1 = a`
* `ix1 = ('i','k')`
* `x2 = b`
* `ix2 = ('k','j')`
* `iy = ('i','j')`.

The set of unique labels in the `ixs = (ix1,ix2)` without the ones in `iy` is then
`l = ('k',)`.
So the output `y` will satisfy
```
∀ ('i','j'): y[i,j] = ∑ₖ a[i,k] * b[k,j]
```
just like we'd expect from a matrix-product.

To find out more about einsum, check out my [nextjournal-article](https://nextjournal.com/under-Peter/julia-summer-of-einsum) or the [numpy-manual](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html).

If we don't specify `iy`, it is constructed from all labels in the `ixs`  that
appear exactly once in alphabetical (if labels are `<:AbstractChar`) or
numerical (if labels are `<:Integer`) order.
If `iy` is given, the specification is evaluated according to the _explicit_ mode in
numpy's `einsum`, otherwise it's evaluated according to the _implicit_ mode.
In the above case, it amounts to the same.

We might instead be interested in the sum of all elements of the matrix product `a*b`
and reduce over all indices, specifying `iy = ()`:
```julia
julia> xs = (rand(2,2), rand(2,2));

julia> ixs = (('i','k'),('k','j'));

julia> iy = ();

julia> einsum(ixs, xs, iy) ≈  sum(a * b)
true
```

`einsum` evaluates the specification `ixs,iy` as a sequence of operations on labels.
The order those operations are evaluated in is given by the ordering of their labels.
E.g. in
```julia
julia> a, b, c = rand(2,2), rand(2,2), rand(2,2);

julia> einsum((('i','j'),('j','h'),('h','k')), (a,b,c), ('i','k')
```
there are two indices that require operations: `'j'` and `'h'` both imply a matrix
product. They are evaluated in alphabetical order, i.e. first `'h'`, then `'j'`,
corresponding to `(a*(b*c))`.

If the order should instead be optimized automatically, use `einsumopt`.
`einsumopt` will calculate the cost of each possible sequence of operations and evaluate
the (possibly nonunique) optimal operations order.
This is currently associated with a rather large overhead,
but an example from physics with unfortunate default evaluation order shows that in some
cases it might still be worth it:
```julia
julia> d = 5; χ = 50; a = randn(χ,χ); b= randn(χ,d,χ); c = randn(d,d,d,d);

julia> @btime einsum((('x','y'), ('x','k','l'), ('y','m','n'), ('k','m','o','p')), ($a, $b, $b, $c), ('l','n','o','p'));
  1.396 s (460 allocations: 2.33 GiB)

julia> @btime einsumopt((('x','y'), ('x','k','l'), ('y','m','n'), ('k','m','o','p')), ($a, $b, $b, $c), ('l','n','o','p'));
  2.579 ms (11789 allocations: 2.00 MiB)

```
although the same effect can be had by choosing the labels appropriately, such that
the best contraction sequence is in alphabetical order:
```julia
julia> @btime einsum((('i','j'), ('i','k','l'), ('j','m','n'), ('k','m','o','p')), ($a, $b, $b, $c), ('l','n','o','p'));
  773.467 μs (339 allocations: 1.54 MiB)
```

(The significant overhead of order optimisation is being worked on).

## Implementation

Under the hood, both `einsum` and `einsumopt` take the input indices `ixs` and `iy`
and translate them to a list of _operators_ which are implemented as concrete subtypes
of the abstract type `EinsumOp`.

One such operator is `Trace(edges)` which holds the labels of the indices that can
be traced over in one operation. An operator together with the tensors and input indices
can be given to the (unexported) `evaluate` function which can dispatch on an
appropriate method.
Trace currently get dispatched to `tensortrace` from the `TensorOperations` package,
whereas an index-reduction like `ij -> i` is represented by `IndexReduction(('i',))`
and dispatches to the built-in `sum` over the appropriate dimension.

The operators currently implemented are:
* `TensorContract`: contract one or more indices from two tensors, e.g. `ijk,jkl -> il`
* `Trace`: contract one or more index-pairs from one tensor, e.g. `iij -> j`
* `StarContract`: contract one or more indices shared between three or more tensors but none of the tensors has duplicate shared indices, e.g. `ij,ik,il -> jkl`
* `MixedStarContract`: contract one or more indices shared between three or more tensors but one or more of the tensors has duplicate shared indices, e.g. `ij,ik,iil -> jkl`
* `Diag`: take the diagonal of one or more indices between multiple tensors but none of the tensors has duplicate shared indices, e.g. `ij,ik -> ij`
* `MixedDiag`: take the diagonal of one or more indices between multiple tensors but one or more of the tensors has duplicate shared indices, e.g. `iij,ik -> ij`
* `IndexReduction`: reduce over one or more indices of a tensor, e.g. `ij -> j`
* `Permutation`: permute the indices of a tensor, e.g. `ijk -> jki`
* `OuterProduct`: take the outer product of one or more tensors, e.g. `ij,kl -> ijkl`
* `Fallback`: all operations not captured by the ones above, e.g. `ij -> iij`

The operators are currently not extendable and have been chosen based on my knowledge on whether there's
a more efficient implementation for an operator than the default.
Thus e.g. I separate e.g. `MixedStarContract` and `MixedDiag` from their respective non-mixed version,
because the nonmixed-version can be nicely calculated using `broadcast`.

`einsum` currently combines operations that are compatible and act on the same tensors.
In e.g. the double trace `ix = ('i','i','k','k')`, `iy = ()`, we can evaluate both
the trace over `'k'` and `'i'` with one function call.

This behaviour is controlled by `iscombineable(op1,op2)` which returns true if two operators can be combined and `combineops(op1,op2)` which is called if two consecutive operations satisfy `iscombineable` and act on the same tensor(s).


## Differences to `numpy.einsum`

Apart from the implementation-details, `OMEinsum.einsum` always returns copies, never views,
whereas `numpy.einsum` might return a view e.g. in the case of a permutation.
`OMEinsum.einsum` also doesn't support the use of ellipsis for unchanged variables, as in
e.g. `numpy.einsum("ij... -> ji...",a)` which would swap the first two axes in `numpy`.


## Contribute

Suggestions and Comments in the _Issues_ are welcome.

## License
MIT License
