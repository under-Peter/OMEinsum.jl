module OMEinsum
export einsum, expandall!
export einsumopt

using TupleTools, Requires, TensorOperations, LinearAlgebra

include("einsumexp.jl")
include("utils.jl")
include("einsum.jl")
include("einsumopt.jl")
include("einorder.jl")
include("einevaluate.jl")
function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cueinsum.jl")
end

include("interfaces.jl")
include("autodiff.jl")
end # module
