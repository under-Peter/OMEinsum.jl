using OMEinsum
using OMEinsum.ContractionOrder
using OMEinsum.ContractionOrder: analyze_contraction, contract_pair!, evaluate_costs, contract_tree!, log2sumexp2
using TropicalNumbers

using Test, Random
@testset "analyze contraction" begin
    incidence_list = IncidenceList(Dict('A' => ['a', 'b', 'k', 'o', 'f'], 'B'=>['a', 'c', 'd', 'm', 'f'], 'C'=>['b', 'c', 'e', 'f'], 'D'=>['e'], 'E'=>['d', 'f']), openedges=['c', 'f', 'o'])
    info = analyze_contraction(incidence_list, 'A', 'B')
    @test Set(info.l1) == Set(['k'])
    @test Set(info.l2) == Set(['m'])
    @test Set(info.l12) == Set(['a'])
    @test Set(info.l01) == Set(['b','o'])
    @test Set(info.l02) == Set(['c', 'd'])
    @test Set(info.l012) == Set(['f'])
end

@testset "tree greedy" begin
    Random.seed!(2)
    incidence_list = IncidenceList(Dict('A' => ['a', 'b'], 'B'=>['a', 'c', 'd'], 'C'=>['b', 'c', 'e', 'f'], 'D'=>['e'], 'E'=>['d', 'f']))
    log2_edge_sizes = Dict([c=>i for (i,c) in enumerate(['a', 'b', 'c', 'd', 'e', 'f'])]...)
    edge_sizes = Dict([c=>(1<<i) for (i,c) in enumerate(['a', 'b', 'c', 'd', 'e', 'f'])]...)
    il = copy(incidence_list)
    contract_pair!(il, 'A', 'B', log2_edge_sizes)
    target = IncidenceList(Dict('A' => ['b', 'c', 'd'], 'C'=>['b', 'c', 'e', 'f'], 'D'=>['e'], 'E'=>['d', 'f']))
    @test il.v2e == target.v2e
    @test length(target.e2v) == length(il.e2v)
    for (k,v) in il.e2v
        @test sort(target.e2v[k]) == sort(v)
    end
    costs = evaluate_costs(MinSpaceOut(), incidence_list, log2_edge_sizes)
    @test costs == Dict(('A', 'B')=>9, ('A', 'C')=>15, ('B','C')=>18, ('B','E')=>10, ('C','D')=>11, ('C', 'E')=>14)
    tree, log2_tcs, log2_scs = tree_greedy(incidence_list, log2_edge_sizes)
    tcs_, scs_ = [], []
    contract_tree!(copy(incidence_list), tree, log2_edge_sizes, tcs_, scs_)
    @test all((log2sumexp2(tcs_), maximum(scs_)) .<= (log2(exp2(10)+exp2(16)+exp2(15)+exp2(9)), 11))
    vertices = ['A', 'B', 'C', 'D', 'E']
    optcode1 = parse_eincode(incidence_list, tree, vertices=vertices)
    @test optcode1 isa OMEinsum.NestedEinsum
    tree2 = OMEinsum.parse_tree(optcode1, vertices)
    @test tree2 == tree

    eincode = ein"ab,acd,bcef,e,df->"
    size_dict = Dict([c=>(1<<i) for (i,c) in enumerate(['a', 'b', 'c', 'd', 'e', 'f'])]...)
    Random.seed!(2)
    optcode2 = optimize_greedy(eincode, size_dict) 
    tc, sc = timespace_complexity(optcode2, edge_sizes)
    @test 16 <= tc <= log2(exp2(10)+exp2(16)+exp2(15)+exp2(9))
    @test sc == 11
    @test optcode1 == optcode2
    eincode3 = ein"(ab,acd),bcef,e,df->"
    Random.seed!(2)
    optcode3 = optimize_greedy(eincode3, size_dict) 
    tc, sc = timespace_complexity(optcode3, edge_sizes)
    @test 16 <= tc <= log2(exp2(10)+exp2(16)+exp2(15)+exp2(9)+1e-8)
end

@testset "fullerene" begin
    function fullerene()
        φ = (1+√5)/2
        res = NTuple{3,Float64}[]
        for (x, y, z) in ((0.0, 1.0, 3φ), (1.0, 2 + φ, 2φ), (φ, 2.0, 2φ + 1.0))
            for (α, β, γ) in ((x,y,z), (y,z,x), (z,x,y))
                for loc in ((α,β,γ), (α,β,-γ), (α,-β,γ), (α,-β,-γ), (-α,β,γ), (-α,β,-γ), (-α,-β,γ), (-α,-β,-γ))
                    if loc ∉ res
                        push!(res, loc)
                    end
                end
            end
        end
        return res
    end

    c60_xy = fullerene()
    c60_edges = [(i,j) for (i,(i2,j2,k2)) in enumerate(c60_xy), (j,(i1,j1,k1)) in enumerate(c60_xy) if i<j && (i2-i1)^2+(j2-j1)^2+(k2-k1)^2 < 5.0]
    code = EinCode((c60_edges..., [(i,) for i=1:60]...), ())
    size_dict = Dict([i=>2 for i in 1:60])
    log2_edge_sizes = Dict([i=>1 for i in 1:60])
    edge_sizes = Dict([i=>2 for i in 1:60])
    tc, sc = timespace_complexity(code, edge_sizes)
    @test tc == 60
    @test sc == 0
    optcode = optimize_greedy(code, size_dict)
    tc2, sc2 = timespace_complexity(optcode, edge_sizes)
    @test sc2 == 10
    xs = vcat([TropicalF64.([-1 1; 1 -1]) for i=1:90], [TropicalF64.([0, 0]) for i=1:60])
    @test OMEinsum.flatten(optcode) == code
    @test OMEinsum.flatten(code) == code
    @test optcode(xs...)[].n == 66
end

@testset "regression test" begin
    code = ein"i->"
    optcode = optimize_greedy(code, Dict('i'=>3))
    @test optcode isa NestedEinsum
    x = randn(3)
    @test optcode(x) ≈ code(x)

    code = ein"i,j->"
    optcode = optimize_greedy(code, Dict('i'=>3, 'j'=>3))
    @test optcode isa NestedEinsum
    x = randn(3)
    y = randn(3)
    @test optcode(x, y) ≈ code(x, y)

    code = ein"ij,jk,kl->ijl"
    optcode = optimize_greedy(code, Dict('i'=>3, 'j'=>3, 'k'=>3, 'l'=>3))
    @test optcode isa NestedEinsum
    a, b, c = [rand(3,3) for i=1:3]
    @test optcode(a, b, c) ≈ code(a, b, c)
end
