# Input (flat)

An einsum specification should be given via the `ein_str` string-literal
or with the `@ein`-macro as e.g.
```@example 2
using OMEinsum
a, b = randn(2, 2), randn(2, 2)

c = ein"ij,jk -> ik"(a,b)
@ein c[i,k] := a[i,j] * b[j,k]
```
where both specifications encode the same operation - a matrix multiplication.
The `ein_str`-literal is parsed directly into an `EinCode` struct that holds
the indices of the input `ixs = (('i','j'),('j','k'))` and output `iy = ('i','k')`
as type parameters, making them accessible at compile time.

The string-literal form gets turned into
```@example 2
c = EinCode((('i','j'),('j','k')),('i','k'))(a,b)
```
Calling an `EinCode`-object gets lowered to
```@example 2
c = einsum(EinCode((('i','j'),('j','k')),('i','k')), (a,b), Dict('i'=>2, 'j'=>2, 'k'=>2))
```
The third argument `size_dict` is a dictionary to specify the dimensions of degree of freedoms, which could also allow to provide dimensions for index-labels that only appear in the output.

In the next step, a singleton-subtype of the abstract type `EinRule` is chosen which is later used for dispatch.
Subtypes of `EinRule` specify the kind of operation and are created in such a way that they allow useful dispatch.
They are defined in `EinRule.jl`.

The possible types are:
- `Identity` - operation is the identity on _one_ tensor, e.g. `ein"ijk -> ijk"`
- `Permutedims` - operation is a permutation of the indices of _one_ tensor, e.g. `ein"ijk -> jki"`
- `Tr` - operation is a trace of _one_ matrix, e.g. `ein"ii ->"`
- `Sum` - operation is a reduction over one or more indices of _one_ tensor, e.g. `ein"ijkl -> il"`
- `SimpleBinaryRule` - operation is a pairwise contraction that can not be reduce by unary operations, e.g. `ein"ijl,jkl-> ikl"`
- `DefaultRule` - default if none of the above match, e.g. `ein"ij,ik,il -> jkl"`

Since `ixs` and `iy` are saved as type-parameters, the operation-matching can happen at compile time.
The operation is chosen using `match_rule(ixs,iy)` by testing all subtypes of `EinRule` in the sequence above (top to bottom) and picking the first match.

This enables us to chose fast BLAS functions for a  matrix multiplication which is also a legal tensor-contraction.

We proceed by calling `einsum(<:EinRule, <:EinCode, xs, size_dict)` which
dispatches on the `EinRule` and the type of `xs` - the latter enables us to dispatch to e.g. cuda-specific routines for certain operations (as done in the `cueinsum.jl` file).

In the case of the matrix-multiplication above, `einsum` calls `*` which can dispatch
to efficient routines for most `Array`-types including `CuArray`.

# Input (Nested)

Whether with the `ein_str` string-literal or the `@ein` macro, nested expressions are mapped to a nested struct.
Consider the example
```@example 2
c = ein"(ij,jk),kl -> il"(a,b,c)
@ein c[i,l] := (a[i,j] * b[j,k]) * c[k,l]
```
which is a simply a product of three matrices evaluated as
two matrix products in sequence.

This is equivalent to
```@example 2
c = ein"ik,kl -> il"(ein"ij,jk -> ik"(a,b),c)
@ein ab[i,k] := a[i,j] * b[j,k]
@ein c[i,l] := ab[i,k] * c[k,l]
```
and is expressed as a nested structure `NestedEinsum`
which contains the `EinCode`s for the intermediate calculations
as well as some logic to assign the correct input and output tensors
to the correct `EinCode`.

`NestedEinsum` has the following definition:
```@example 2
struct NestedEinsum
    args
    eins
end
```
`args` holds the arguments to that `EinCode` which can either be a integer to label a tensor or a `NestedEinsum` itself.
The labeling works such that the `i`th input is represented by the number `i`.

Upon application to tensors, a `NestedEinsum` evaluates its arguments.
If the argument is an integer `i`, the `i`th provided tensor is chosen,
otherwise the `NestedEinsum` is evaluated.

To make it more concrete, consider the `NestedEinsum` for the expression above, where for easier reading the type signatures were removed and the `EinCode`-structs were replaced by `ein`-string literals.
```@example 2
ein"(ij,jk),kl -> il"
```
Evaluating this expression with three arguments leads to the inner `NestedEinsum` to be evaluated first with the first and second argument and the specifiation `ein"ij,jk -> ik"`. Then the result of that is given
as the first argument to `ein"ik,kl -> il"` with the third argument as the second input.

To improve understanding, you might replace the integers with `getindex` operations in your head
```julia
ein"(ij,jk),kl -> il"(xs...)
⇒ NestedEinsum{...}((NestedEinsum{...}((xs[1], xs[2]), ein"ij,jk -> ik"), xs[3]), ein"ik,kl -> il")
```
and finally turn it into
```julia
ein"(ij,jk),kl -> il"(xs...)
⇒ ein"ik,kl -> il"(ein"ij,jk -> ik"(xs[1], xs[2]), xs[3])
```