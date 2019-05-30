module OMEinsum
export einsum, expandall!

using TupleTools

include("einsum.jl")
include("autodiff.jl")

end # module
