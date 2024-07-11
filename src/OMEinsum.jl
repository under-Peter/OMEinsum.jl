module OMEinsum

using TupleTools, Base.Cartesian, LinearAlgebra
using BatchedRoutines
using OMEinsumContractionOrders
using AbstractTrees
import LinearAlgebra: BlasFloat

export @ein_str, @ein, @ein!, ein, @optein_str
export einsum!, einsum, dynamic_einsum
export EinCode, EinIndexer, EinArray, DynamicEinCode, StaticEinCode, AbstractEinsum, NestedEinsum, SlicedEinsum, DynamicNestedEinsum, StaticNestedEinsum
export getiyv, getixsv, uniquelabels, labeltype
export flop
export loop_einsum, loop_einsum!, allow_loops
export asarray, asscalar
export cost_and_gradient

# re-export the functions in OMEinsumContractionOrders
export CodeOptimizer, CodeSimplifier,
    KaHyParBipartite, GreedyMethod, TreeSA, SABipartite,
    MinSpaceDiff, MinSpaceOut,
    MergeGreedy, MergeVectors,
    uniformsize,
    optimize_code, optimize_permute,
    # time space complexity
    peak_memory, timespace_complexity, timespacereadwrite_complexity, flop, contraction_complexity,
    # file io
    writejson, readjson,
    label_elimination_order

include("Core.jl")
include("loop_einsum.jl")
include("utils.jl")

include("unaryrules.jl")
include("binaryrules.jl")
include("matchrule.jl")
include("einsum.jl")

include("interfaces.jl")
include("einsequence.jl")
include("slicing.jl")
include("autodiff.jl")
include("bp.jl")

include("contractionorder.jl")

include("deprecation.jl")
end # module
