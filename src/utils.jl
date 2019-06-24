Base.setdiff(t::Tuple, b) = setdiff!(collect(t), b)
Base.unique(t::Tuple) = unique!(collect(t))
