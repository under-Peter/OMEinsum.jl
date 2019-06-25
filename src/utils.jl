asarray(x::Number) = fill(x, ())
asarray(x::AbstractArray) = x

tsetdiff(t::Tuple, b) = setdiff!(collect(t), b)
tunique(t::Tuple) = unique!(collect(t))
