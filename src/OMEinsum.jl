module OMEinsum

using TupleTools, Base.Cartesian, LinearAlgebra
using BatchedRoutines
using OMEinsumContractionOrders
using AbstractTrees
import LinearAlgebra: BlasFloat

export @ein_str, @ein, ein
export einsum, dynamic_einsum
export EinCode, EinIndexer, EinArray, DynamicEinCode, StaticEinCode, AbstractEinsum, NestedEinsum, SlicedEinsum
export getiyv, getixsv, uniquelabels, labeltype
export timespace_complexity, timespacereadwrite_complexity
export flop
export loop_einsum, loop_einsum!, allow_loops
export asarray, asscalar

# re-export the functions in OMEinsumContractionOrders
export CodeOptimizer, CodeSimplifier,
    KaHyParBipartite, GreedyMethod, TreeSA, SABipartite,
    MinSpaceDiff, MinSpaceOut,
    MergeGreedy, MergeVectors,
    uniformsize,
    optimize_code, optimize_permute,
    # time space complexity
    peak_memory, timespace_complexity, timespacereadwrite_complexity, flop,
    # file io
    writejson, readjson,
    label_elimination_order

const CuBlasFloat = Union{BlasFloat, Float16, ComplexF16}

include("Core.jl")
include("loop_einsum.jl")
include("utils.jl")

include("unaryrules.jl")
include("binaryrules.jl")

using Requires
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cueinsum.jl")
end

include("interfaces.jl")
include("einsequence.jl")
include("slicing.jl")
include("autodiff.jl")

include("contractionorder.jl")

include("deprecation.jl")
end # module
