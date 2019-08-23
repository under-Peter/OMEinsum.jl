# Input (flat)

An einsum specification should be given via the `ein_str` string-literal
or with the `@ein`-macro as e.g.
```julia
julia> c = ein"ij,jk -> ik"(a,b)
julia> @ein c[i,k] := a[i,j] * b[j,k]
```
where both specifications encode the same operation - a matrix multiplication.
The `ein_str`-literal is parsed directly into an `EinCode` struct that holds
the indices of the input `ixs = (('i','j'),('j','k'))` and output `iy = ('i','k')`
as type parameters, thus making them accessible at compile time.

The string-literal form thus gets turned into
```julia
julia> c = EinCode((('i','j'),('j','k')),('i','k'))(a,b)
```
Calling an `EinCode`-object gets lowered to
```julia
julia> c = einsum(EinCode((('i','j'),('j','k')),('i','k')), (a,b), size_dict = nothing)
```
where `nothing` is the default argument for the (as of yet not used during specification)
`size_dict`, which could allow to provide dimensions for index-labels that only appear
in the output.

In the next step, a singleton-subtype of the abstract type `EinRule` is chosen which is later used for dispatch.
Subtypes of `EinRule` specify the kind of operation and are created in such a way that they allow useful dispatch.
They are defined in `EinRule.jl`.

The possible types are:
- `Identity` - operation is the identity on _one_ tensor, e.g. `ein"ijk -> ijk"`
- `MatMul` - operation is a matrix multiplication of _two_ matrices, possibly with permutations of inputs and/or outputs, e.g. `ein"ij,kj -> ik"`
- `Permutedims` - operation is a permutation of the indices of _one_ tensor, e.g. `ein"ijk -> jki"`
- `Hadamard` - operation is a hadamard-product of arbitrary many tensors, e.g. `ein"ij,ij,ij -> ij"`
- `Tr` - operation is a trace of _one_ matrix, e.g. `ein"ii ->"`
- `PTrace` - operation is a partial trace of _one_ tensor, e.g. `ein"iij -> j"`
- `Sum` - operation is a reduction over one or more indices of _one_ tensor, e.g. `ein"ijkl -> il"`
- `PairWise` - operation is a tensor-contraction over arbitrary many tensors, e.g. `ein"ijk,kl,lmn,no -> ijmo"`
- `DefaultRule` - default if none of the above match, e.g. `ein"ij,ik,il -> jkl"`

Since `ixs` and `iy` are saved as type-parameters, the operation-matching can happen at compile time.
The operation is chosen using `match_rule(ixs,iy)` by testing all subtypes of `EinRule` in the sequence above (top to bottom) and picking the first match.

This enables us to chose `MatMul` for a  matrix multiplication which is also a legal tensor-contraction, i.e. a `PairWise`, assuming that we can have a lower-overhead implementation for `MatMul` than `PairWise`.

We proceed by calling `einsum(<:EinRule, <:EinCode, xs, size_dict)` which
dispatches on the `EinRule` and the type of `xs` - the latter enables us to dispatch to e.g. cuda-specific routines for certain operations (as done in the `cueinsum.jl` file).

In the case of the matrix-multiplication above, `einsum` calls `*` which can dispatch
to efficient routines for most `Array`-types including `CuArray`.

# Input (Nested)

Whether with the `ein_str` string-literal or the `@ein` macro, nested expressions are mapped to a nested struct.
Consider the example
```julia
julia> c = ein"(ij,jk),kl -> il"(a,b,c)
julia> @ein c[i,l] := (a[i,j] * b[j,k]) * c[k,l]
```
which is a simply a product of three matrices evaluated as
two matrix products in sequence.

This is equivalent to
```julia
julia> c = ein"ik,kl -> il"(ein"ij,jk -> ik"(a,b),c)
julia> @ein ab[i,k] := a[i,j] * b[j,k]
julia> @ein c[i,l] := ab[i,k] * c[k,l]
```
and is expressed as a nested structure `NestedEinsumStable`
which contains the `EinCode`s for the intermediate calculations
as well as some logic to assign the correct input and output tensors
to the correct `EinCode`.

`NestedEinsumStable` has the following definition:
```julia
struct NestedEinsumStable{T,S,N}
    args::S
    eins::T
end
```
where the `eins`-field contains an `EinCode` of `N` arguments and
`args` holds the arguments to that `EinCode` which can either be a integer to label a tensor or a `NestedEinsumStable` itself.
The labeling works such that the `i`th input is represented by the number `i`.

Upon application to tensors, a `NestedEinsumStable` evaluates its arguments.
If the argument is an integer `i`, the `i`th provided tensor is chosen,
otherwise the `NestedEinsumStable` is evaluated.

To make it more concrete, consider the `NestedEinsumStable` for the expression above, where for easier reading the type signatures were removed and the `EinCode`-structs were replaced by `ein`-string literals.
```julia
julia> ein"(ij,jk),kl -> il"
 NestedEinsumStable{...}((NestedEinsumStable{...}((1, 2), ein"ij,jk -> ik"), 3), ein"ik,kl -> il")
```
Evaluating this expression with three arguments leads to the inner `NestedEinsumStable` to be evaluated first with the first and second argument and the specifiation `ein"ij,jk -> ik"`. Then the result of that is given
as the first argument to `ein"ik,kl -> il"` with the third argument as the second input.

To improve understanding, you might replace the integers with `getindex` operations in your head
```julia
ein"(ij,jk),kl -> il"(xs...)
⇒ NestedEinsumStable{...}((NestedEinsumStable{...}((xs[1], xs[2]), ein"ij,jk -> ik"), xs[3]), ein"ik,kl -> il")
```
and finally turn it into
```julia
ein"(ij,jk),kl -> il"(xs...)
⇒ ein"ik,kl -> il"(ein"ij,jk -> ik"(xs[1], xs[2]), xs[3])
```
