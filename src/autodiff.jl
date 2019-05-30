using Zygote

function einsum_grad(ixs, xs, iy, y, i)
    T = mapreduce(eltype, promote_type, xs)
    T = promote_type(T, eltype(y))
    nixs = TupleTools.insertat(ixs, i, (iy,))
    nxs  = TupleTools.insertat( xs, i, ( y,))
    niy = ixs[i]
    ny = zeros(T, size(xs[i])...)
    einsum!(nixs, nxs, niy, ny)
    conj!(ny)
end

@Zygote.adjoint function einsum!(ixs, xs::NTuple{N,T where T}, iy, y) where N
    einsum!(ixs, xs, iy, y)
    return y, dy -> let cdy = conj!(copy(dy))
                (
                    nothing,
                    ntuple(i -> einsum_grad(ixs, xs, iy, cdy, i), N),
                    nothing,
                    nothing
                )
            end
end


function bpcheck(f, args...; η = 1e-5, verbose = false)
    g = gradient(f, args...)
    dy_ref = 0
    for x in g
        x === nothing && continue
        x isa Tuple && (dy_ref += η * mapreduce(y -> y == nothing ? 0 : sum(abs2,y), +, x))
        x isa AbstractArray && (dy_ref += η * sum(abs2,x))
    end
    dy = f(args...) - f([gi == nothing ? arg : arg .- η .* gi for (arg, gi) in zip(args,g)]...)

    verbose && @show dy
    verbose && @show dy_ref

    isapprox(dy, dy_ref, rtol=1e-2, atol=1e-8)
end
