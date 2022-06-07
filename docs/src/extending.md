# Extending OMEinsum

Adding a new subtype of `EinRule` is bothersome - the list of rules
that's considered needs to be fix and thus one has to change the code before
`using` OMEinsum. A limitation due to liberal use of `generated` functions.
If a useful rule is found, we might add it to the package itself though so feel free to reach out.

Extending `einsum` for certain array-types on the other hands is easy,
since we use the usual dispatch mechanism.
Consider e.g. adding a special operator for index-reductions of a `Diagonal`-operator.

First, we need to add a method for the `asarray`-function that ensures that we return 0-dimensional arrays for operations.

```@example 1
using OMEinsum, LinearAlgebra

OMEinsum.asarray(a::Number, ::Diagonal) = fill(a,())
```

Now reducing over indices already works but it uses the `sum` function
which does not specialize on `Diagonal`:
```@example 1
ein"ij -> "(Diagonal([1,2,3]))
```

we can do better by overloading the unary rule `einsum(::Sum, ixs, iy, ::Tuple{<:Diagonal}, <:Any)`:
```@example 1
function OMEinsum.einsum(::OMEinsum.Sum, ixs, iy, xs::Tuple{<:Diagonal}, size_dict::Dict)
    length(iy) == 1 && return diag(xs[1])
    return sum(diag(xs[1]))
end
```

where we use that the indices `iy` and `ixs` have already been checked in `match_rule`.
We now get our more efficient implementation when we call any of the below:
```@example 1
ein"ij -> i"(Diagonal([1,2,3]))
```

```@example 1
ein"ij -> j"(Diagonal([1,2,3]))
```

```@example 1
ein"ij -> "(Diagonal([1,2,3]))
```

(To make sure the custom implementation is called, you can add a `print`-statement to the method for `Diagonal`)
