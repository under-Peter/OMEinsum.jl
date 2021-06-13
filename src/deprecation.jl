@deprecate einsumexp(args...) loop_einsum(args...)
@deprecate einsumexp!(args...) loop_einsum!(args...)

@deprecate einsumopt einsum

# deprecated manually
function Iterators.flatten(c::Union{NestedEinsum,EinCode})
    @warn "`Iterators.flatten(eincode)` has been deprecated, use `OMEinsum.flatten` instead."
    flatten(c)
end
