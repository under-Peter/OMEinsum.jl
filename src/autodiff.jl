using Zygote

@Zygote.nograd parseeinsumsstring

@doc raw"
    einsum_grad(ixs, xs, iy, y, i)
return gradient w.r.t the `i`th tensor in `xs`
"
function einsum_grad(ixs, xs, iy, y, i)
    T = mapreduce(eltype, promote_type, xs)
    T = promote_type(T, eltype(y))
    nixs = TupleTools.insertat(ixs, i, (iy,))
    nxs  = TupleTools.insertat( xs, i, ( y,))
    niy = ixs[i]
    ntmp = Tuple(i for i in unique(niy) if any(x -> i in x, nixs))
    tmp = einsum(nixs, nxs, ntmp)
    ny = zeros(T, size(xs[i])...)
    einsumexp!((ntmp,), (tmp,), niy, ny)
    conj!(ny)
end

@Zygote.adjoint function einsum(ixs, xs::NTuple{N,T where T}, iy) where N
    y = einsum(ixs, xs, iy)
    return y, dy -> let cdy = map(conj,dy)
                (
                    nothing,
                    ntuple(i -> einsum_grad(ixs, xs, iy, cdy, i), N),
                    nothing,
                    nothing
                )
            end
end


@doc raw"
    bpcheck(f, args...; η = 1e-5, verbose=false)
returns a `Bool` indicating whether Zygote calculates the gradient of `f(args...) -> scalar`
correctly using the relation `f(x - ηg) ≈ f(x) - η|g|²`.
If `verbose=true`, print `f(x) - f(x - ηg)`and `η|g|²`.
"
function bpcheck(f, args...; η = 1e-5, verbose = false)
    g = gradient(f, args...)
    all(==(nothing), g) && error()
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

@Zygote.nograd outputtensor
