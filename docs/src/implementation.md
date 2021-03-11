# Implementations

## Identity
To test whether a specification `ixs,iy` is the identity, it is checked whether
`ixs` is made up of _one_ tuple of index-labels that is equal to `iy` _and_
that all index-labels in `iy` are unique - the latter to distuingish identity
from e.g. projection to the diagonal like `ein"ii -> ii"`.

The identity operation simply returns the first (and only) tensor argument to `einsum`.

## Index-Permutation

A specification `ixs,iy` is an index-permutation if `ixs` is a tuple containing
one tuple of index-labels that are all unique and are a permutation of the labels
in `iy`.

Index-permutation is implemented with `permutedims` and a permutation that's calculated
at runtime.

## Trace

A specification `ixs, iy` is a trace if `iy` is empty and `ixs` contains one
2-tuple containing the same index-label twice.

A trace dispatches to the `LinearAlgebra.tr` although the result is wrapped in
a 0-dimensional array for type stability since all `einsum` return `AbstractArray`s.

## Sum

A specification `ixs,iy` is a sum or a reduction over indices if all indices in `iy`
are unique and contained in the only tuple in `ixs` that additionally contains
unique labels (that are reduced over).

Index-reductions are implemented using `Base.sum` and `Base.dropdims` - the latter
to remove the singleton-dimensions left over after summing over a dimension.

## SimpleBinaryRule
The contraction between two tensors with the following restriction
* a tensor can not be simplified by unary rules, e.g. `iij,jk,ik` is not valid, the first index can be simplified to `ij` using the unary rule `iij->ij`.
* no multi-edge

A complete list of rules are
* ein",->"
* ein",k->k"
* ein"i,->i"
* ein"j,j->"
* ein"i,k->ik" and ein"i,k->ki",
* ein"j,jk->k" and ein"j,kj->k"
* ein"ji,j->i" and ein"ij,j->i"
* ein"ji,jk->ik" and its index permutations (within a tensor)
* ein"l,l->l"
* ein"l,kl->kl"
* ein"il,->il"
* ein"jl,jl->"
* ein"il,kl->ikl" and ein"il,kl->kil",
* ein"jl,jkl->kl" and ein"jl,kjl->kl"
* ein"jil,jl->il" and ein"ijl,jl->il"
* ein"jil,jkl->ikl" and its index permutations (within a tensor, except the batch dimension)

Here, the batch dimension always appears as the last dimension.

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

## Debugging

Calling `allow_loops(false)` will cause an error to be pinted when if the 
fallback `loop_einsum` is used. This is an `@error` which does not interrupt execution. 

Alternatively, a log of all methods used can be saved using `@debug` logging macro. 
This is switched off by default, but can be printed by setting `ENV["JULIA_DEBUG"] = "all"`.
