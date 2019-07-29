using Zygote

@doc raw"
    einsum_grad(ixs, xs, iy, y, i)
return gradient w.r.t the `i`th tensor in `xs`
"
function einsum_grad(::EinCode{ixs, iy}, xs, size_dict, cdy, i) where {ixs, iy}
    T = mapreduce(eltype, promote_type, xs)
    T = promote_type(T, eltype(cdy))
    nixs = TupleTools.insertat(ixs, i, (iy,))
    nxs  = TupleTools.insertat( xs, i, (cdy,))
    niy = ixs[i]
    conj!(einsum(EinCode(nixs, niy), nxs, size_dict))
end

@Zygote.adjoint function einsum(code::EinCode{ixs, iy}, xs::NTuple{N,T where T}, size_dict::IndexSize) where {N, ixs, iy}
    y = einsum(code, xs, size_dict)
    return y, dy -> let cdy = map(conj,dy)
                (
                    nothing,
                    ntuple(i -> einsum_grad(code, xs, size_dict, cdy, i), N),
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

@Zygote.nograd get_size_dict
