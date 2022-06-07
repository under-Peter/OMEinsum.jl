module OMEinsum

using TupleTools, Base.Cartesian, LinearAlgebra
using BatchedRoutines
import LinearAlgebra: BlasFloat

export @ein_str, @ein, ein
export einsum, dynamic_einsum
export EinCode, EinIndexer, EinArray, DynamicEinCode, StaticEinCode, AbstractEinsum, NestedEinsum
export getiyv, getixsv, uniquelabels, labeltype
export timespace_complexity, timespacereadwrite_complexity
export flop
export loop_einsum, loop_einsum!, allow_loops
export asarray, asscalar

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
include("autodiff.jl")

include("contractionorder/contractionorder.jl")

include("deprecation.jl")
end # module
