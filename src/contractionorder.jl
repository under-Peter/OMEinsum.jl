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

function rawcode(code::EinCode)
    OMEinsumContractionOrders.EinCode(getixsv(code), getiyv(code))
end
function rawcode(code::NestedEinsum)
    if isleaf(code)
        OMEinsumContractionOrders.NestedEinsum{labeltype(code)}(code.tensorindex)
    else
        OMEinsumContractionOrders.NestedEinsum(rawcode.(code.args), rawcode(code.eins))
    end
end
function rawcode(code::SlicedEinsum)
    OMEinsumContractionOrders.SlicedEinsum(code.slicing, rawcode(code.eins))
end

function OMEinsumContractionOrders.optimize_code(code::AbstractEinsum, size_dict::Dict, optimizer::CodeOptimizer, simplifier=nothing, permute::Bool=true)
    decorate(optimize_code(rawcode(code), size_dict, optimizer, simplifier, permute))
end
OMEinsumContractionOrders.simplify_code(code::AbstractEinsum, size_dict, simplifier::CodeSimplifier) = decorate(simplify_code(rawcode(code), size_dict, simplifier))
OMEinsumContractionOrders.peak_memory(code::AbstractEinsum, size_dict::Dict) = peak_memory(rawcode(code), size_dict)
OMEinsumContractionOrders.timespacereadwrite_complexity(code::AbstractEinsum, size_dict) = timespacereadwrite_complexity(rawcode(code), size_dict)

# TODO: figure out where is this function
OMEinsumContractionOrders.uniformsize(code::AbstractEinsum, size) = Dict([l=>size for l in uniquelabels(code)])
