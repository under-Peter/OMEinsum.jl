function decorate(code::OMEinsumContractionOrders.EinCode)
    DynamicEinCode(code.ixs, code.iy)
end
function decorate(code::OMEinsumContractionOrders.NestedEinsum{LT}) where LT
    if OMEinsumContractionOrders.isleaf(code)
        DynamicNestedEinsum{LT}(code.tensorindex)
    else
        DynamicNestedEinsum(decorate.(code.args), decorate(code.eins))
    end
end
function decorate(code::OMEinsumContractionOrders.SlicedEinsum)
    SlicedEinsum(code.slicing, decorate(code.eins))
end

function rawcode(code::EinCode)
    OMEinsumContractionOrders.EinCode(getixsv(code), getiyv(code))
end
function rawcode(code::NestedEinsum{LT}) where LT
    if isleaf(code)
        OMEinsumContractionOrders.NestedEinsum{LT}(tensorindex(code))
    else
        OMEinsumContractionOrders.NestedEinsum([rawcode(s) for s in siblings(code)], rawcode(rootcode(code)))
    end
end
function rawcode(code::SlicedEinsum)
    OMEinsumContractionOrders.SlicedEinsum(code.slicing, rawcode(code.eins))
end
rawcode(code::AbstractEinsum) = rawcode(code)

function OMEinsumContractionOrders.optimize_code(code::AbstractEinsum, size_dict::Dict, optimizer::CodeOptimizer; simplifier=nothing, permute::Bool=true, slicer=nothing)
    decorate(optimize_code(rawcode(code), size_dict, optimizer; simplifier, permute, slicer))
end
function OMEinsumContractionOrders.slice_code(code::NestedEinsum{LT}, size_dict::Dict, slicer::CodeSlicer) where LT
    decorate(OMEinsumContractionOrders.slice_code(rawcode(code), size_dict, slicer))
end
OMEinsumContractionOrders.optimize_permute(code::AbstractEinsum) = decorate(optimize_permute(rawcode(code)))
OMEinsumContractionOrders.peak_memory(code::AbstractEinsum, size_dict::Dict) = peak_memory(rawcode(code), size_dict)
OMEinsumContractionOrders.flop(code::AbstractEinsum, size_dict::Dict) = flop(rawcode(code), size_dict)
OMEinsumContractionOrders.contraction_complexity(code::AbstractEinsum, size_dict) = contraction_complexity(rawcode(code), size_dict)

OMEinsumContractionOrders.uniformsize(code::AbstractEinsum, size) = Dict([l=>size for l in uniquelabels(code)])
OMEinsumContractionOrders.label_elimination_order(code::AbstractEinsum) = label_elimination_order(rawcode(code))

OMEinsumContractionOrders.viz_eins(code::AbstractEinsum, args...; kwargs...) = viz_eins(rawcode(code), args...; kwargs...)
OMEinsumContractionOrders.viz_contraction(code::AbstractEinsum, args...; kwargs...) = viz_contraction(rawcode(code), args...; kwargs...)

# save load
function writejson(filename::AbstractString, ne::Union{NestedEinsum, SlicedEinsum})
    OMEinsumContractionOrders.writejson(filename, rawcode(ne))
end
readjson(filename::AbstractString) = decorate(OMEinsumContractionOrders.readjson(filename))
