module OMEinsum
export einsum, expandall!
export meinsumopt

using TupleTools

include("einsum.jl")
include("autodiff.jl")
include("einorder.jl")

end # module
