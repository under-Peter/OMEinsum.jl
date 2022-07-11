function decorate(code::OMEinsumContractionOrders.EinCode)
    DynamicEinCode(code.ixs, code.iy)
end
function decorate(code::OMEinsumContractionOrders.NestedEinsum{LT}) where LT
    if OMEinsumContractionOrders.isleaf(code)
        NestedEinsum{DynamicEinCode{LT}}(code.tensorindex)
    else
        NestedEinsum(decorate.(code.args), decorate(code.eins))
    end
end
function decorate(code::OMEinsumContractionOrders.SlicedEinsum)
    SlicedEinsum(code.slicing, decorate(code.eins))
end

function rawcode(::Type{LT}, code::EinCode) where LT
    OMEinsumContractionOrders.EinCode(getixsv(code), getiyv(code))
end
function rawcode(::Type{LT}, code::NestedEinsum) where LT
    if isleaf(code)
        OMEinsumContractionOrders.NestedEinsum{LT}(code.tensorindex)
    else
        OMEinsumContractionOrders.NestedEinsum(rawcode.(LT, code.args), rawcode(LT, code.eins))
    end
end
function rawcode(::Type{LT}, code::SlicedEinsum) where LT
    OMEinsumContractionOrders.SlicedEinsum(code.slicing, rawcode(LT, code.eins))
end
rawcode(code::AbstractEinsum) = rawcode(labeltype(code), code)

function OMEinsumContractionOrders.optimize_code(code::AbstractEinsum, size_dict::Dict, optimizer::CodeOptimizer, simplifier=nothing, permute::Bool=true)
    decorate(optimize_code(rawcode(code), size_dict, optimizer, simplifier, permute))
end
OMEinsumContractionOrders.optimize_permute(code::AbstractEinsum) = decorate(optimize_permute(rawcode(code)))
OMEinsumContractionOrders.peak_memory(code::AbstractEinsum, size_dict::Dict) = peak_memory(rawcode(code), size_dict)
OMEinsumContractionOrders.flop(code::AbstractEinsum, size_dict::Dict) = flop(rawcode(code), size_dict)
OMEinsumContractionOrders.timespacereadwrite_complexity(code::AbstractEinsum, size_dict) = timespacereadwrite_complexity(rawcode(code), size_dict)

OMEinsumContractionOrders.uniformsize(code::AbstractEinsum, size) = Dict([l=>size for l in uniquelabels(code)])
OMEinsumContractionOrders.label_elimination_order(code::AbstractEinsum) = label_elimination_order(rawcode(code))

# save load
function writejson(filename::AbstractString, ne::Union{NestedEinsum, SlicedEinsum})
    OMEinsumContractionOrders.writejson(filename, rawcode(ne))
end
readjson(filename::AbstractString) = decorate(OMEinsumContractionOrders.readjson(filename))