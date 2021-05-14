module OMEinsum
export einsum
export einsumopt

using TupleTools, LinearAlgebra
using BatchedRoutines
import LinearAlgebra: BlasFloat
const CuBlasFloat = Union{BlasFloat, Float16, ComplexF16}

include("Core.jl")
include("loop_einsum.jl")
include("utils.jl")

@nospecialize
include("unaryrules.jl")
include("binaryrules.jl")
@specialize

using Requires
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cueinsum.jl")
end

include("interfaces.jl")
include("einsequence.jl")
include("autodiff.jl")

include("deprecation.jl")
end # module
