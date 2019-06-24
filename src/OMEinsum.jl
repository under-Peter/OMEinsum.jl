module OMEinsum
export einsum, expandall!
export einsumopt

using TupleTools, Requires

include("utils.jl")
include("einsum.jl")
include("autodiff.jl")
include("einorder.jl")
include("einevaluate.jl")

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cueinsum.jl")

end # module
