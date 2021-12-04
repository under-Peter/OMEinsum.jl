@deprecate einsumexp(args...) loop_einsum(args...)
@deprecate einsumexp!(args...) loop_einsum!(args...)

@deprecate einsumopt einsum

# deprecated manually
function Iterators.flatten(c::Union{NestedEinsum,EinCode})
    @warn "`Iterators.flatten(eincode)` has been deprecated, use `OMEinsum.flatten` instead."
    flatten(c)
end

@deprecate dynamic_einsum(ixs, xs, iy; size_info=nothing) einsum(DynamicEinCode(ixs, iy), xs; size_info=size_info)
@deprecate dynamic_einsum(code::EinCode, xs; size_info=nothing) code(xs...; size_info=size_info)
@deprecate dynamic_einsum(code::NestedEinsum, xs; size_info=nothing) code(xs...; size_info=size_info)

@deprecate collect_ixs getixsv