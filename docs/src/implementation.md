# Implementations

## Identity
To test whether a specification `ixs,iy` is the identity, it is checked whether
`ixs` is made up of _one_ tuple of index-labels that is equal to `iy` _and_
that all index-labels in `iy` are unique - the latter to distuingish identity
from e.g. projection to the diagonal like `ein"ii -> ii"`.

The identity operation simply returns the first (and only) tensor argument to `einsum`.

## Matrix Multiplication
A specification `ixs,iy` is a matrix multiplication if `ixs` consists of two 2-tuples
that share _one_ index-label and `iy` is a permutation of the two-nonshared index-labels
of the `ixs`.

Matrix multiplication uses a generated function to return a matrix-product with at most
one application of `transpose`, such that e.g. `ein"ij,jk -> ik"(a,b)` returns `:(a * b)`
and `ein"ij,kj -> ki"(a,b)` returns `:(b * transpose(a))`.

## Index-Permutation

A specification `ixs,iy` is an index-permutation if `ixs` is a tuple containing
one tuple of index-labels that are all unique and are a permutation of the labels
in `iy`.

Index-permutation is implemented with `permutedims` and a permutation that's calculated
at runtime.

## Hadamard

A specification `ixs, iy` is a hadamard-product if `ixs` is a tuple that contains
copies of `iy` and nothing else and `iy` contains no duplicates.

The hadamard-product is implemented by broadcasting `*` over the tensors.
If some of the index-labels in `ixs` are permutations of `iy`, we found that
doing the permutation and then broadcasting `*` had worse performance than the
fallback implementation below - we are thus rather strict about what is a
hadamard-product.

## Trace

A specification `ixs, iy` is a trace if `iy` is empty and `ixs` contains one
2-tuple containing the same index-label twice.

A trace dispatches to the `LinearAlgebra.tr` although the result is wrapped in
a 0-dimensional array for type stability since all `einsum` return `AbstractArray`s.

## Partial Trace

A specification `ixs, iy` is a partial trace if `iy` contains no duplicates and
`ixs` is a tuple containing one tuple of index-labels that contains all index-labels
in `iy` plus pairs of labels not in `iy` in arbitrary order.

Partial traces are implemented using `TensorOperations.jl` for regular `AbstractArray`s
and with the Fallback-option (see below) for `CuArray`s, since at this point
`TensorOperations.jl` lacks full GPU support.

## Sum

A specification `ixs,iy` is a sum or a reduction over indices if all indices in `iy`
are unique and contained in the only tuple in `ixs` that additionally contains
unique labels (that are reduced over).

Index-reductions are implemented using `Base.sum` and `Base.dropdims` - the latter
to remove the singleton-dimensions left over after summing over a dimension.

## Tensor-Contractions (PairWise)

A specification `ixs,iy` corresponds to tensor-contractions if all all labels in `iy`
are unique and all indices appear exactly twice in `ixs` and `iy`.

Such operations can be dispatched to `TensorOperations.jl` and are evaluated using
the `@tensoropt` macro which chooses a suitable contraction order for the problem
for all `AbstractArray` except `CuArray`s which are implemented using the Fallback.

## Fallback

The fallback is called for any specification that does not satisfy the criteria
outlined above.

The dispatch calls `loop_einsum` which is defined in `loop_einsum.jl`.

`loop_einsum` is based on the `EinArray`-struct.
An `EinArray` is a subtype of `AbstractArray` that represents an intermediate
step in a general einsum-expression _before_ reductions remove indices.
Consider a specification `ixs,iy` - the `EinArray` for that specification is
the array with an index for each (distinct) label in `ixs` and `iy`.
As an example, in `ein"ij,ik,il -> jkl"(a,b,c)`, the distinct labels are `(i,j,k,l)`
and the corresponding `EinArray` `einarr` would be a rank-4 tensor with an index each for
each distinct label.

If an entry of `einarr` is requested, e.g. `einarr[i₁,j₁,k₁,l₁]`, it's values is lazily
constructed as `einarr[i₁,j₁,k₁,l₁] = a[i₁,j₁]*a[i₁,k₁]*a[i₁,l₁]` upon access - the lazy evaluation avoids constructing the whole array.

To get to the final result, we reduce over the dimensions that are missing in
the output. By first allocating an array of the correct size, we can fill it
up with the entries of the `EinArray` which are calculated on the fly,
avoiding the allocation of the intermediate result.

Thus effectively we split an operation like `ein"ij,ik,il -> jkl"(a,b,c)` into
two piece: `einarr = ein"ij,ik,il -> ijkl"(a,b,c)` and `ein"ijkl -> jkl"(einarr)`
but treat the first operation as a lazy one - this way we can use `mapreduce(identity, +)`
over the dimensions we want to remove which is implemented efficiently for both
regular `Array`s and `CuArray`s.
