using OMEinsum

function random_contract(D=8)
    Ds = [rand(1:D) for _=1:3]
    T = Ds[1] + Ds[2]
    ixs = Tuple.((rand(1:T, Ds[1]), rand(1:T, Ds[2])))
    iy = Tuple(rand(ixs[1] ∪ ixs[2], Ds[3]))
    code = EinCode(ixs, iy)
    code(randn(fill(2, Ds[1])...), randn(fill(2, Ds[2])...))
end

@time for i=1:100
    @show i
    random_contract(8)
end