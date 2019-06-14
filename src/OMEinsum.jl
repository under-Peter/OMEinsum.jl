module OMEinsum
export einsum, expandall!
export einsumopt

using TupleTools

include("einsum.jl")
include("autodiff.jl")
include("einorder.jl")
include("einevaluate.jl")

end # module
