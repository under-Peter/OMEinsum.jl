using OMEinsum, OMEinsumContractionOrders
using TropicalNumbers
using Test, Random

@testset "tree greedy" begin
    Random.seed!(2)
    #incidence_list = IncidenceList(Dict('A' => ['a', 'b'], 'B'=>['a', 'c', 'd'], 'C'=>['b', 'c', 'e', 'f'], 'D'=>['e'], 'E'=>['d', 'f']))
    #log2_edge_sizes = Dict([c=>i for (i,c) in enumerate(['a', 'b', 'c', 'd', 'e', 'f'])]...)
    edge_sizes = Dict([c=>(1<<i) for (i,c) in enumerate(['a', 'b', 'c', 'd', 'e', 'f'])]...)
    eincode = ein"ab,acd,bcef,e,df->"
    size_dict = Dict([c=>(1<<i) for (i,c) in enumerate(['a', 'b', 'c', 'd', 'e', 'f'])]...)
    Random.seed!(2)
    optcode2 = optimize_code(eincode, size_dict, GreedyMethod()) 
    tc, sc = contraction_complexity(optcode2, edge_sizes)
    # test flop
    @test tc ≈ log2(flop(optcode2, edge_sizes))
    @test flop(ein"i->", Dict('i'=>4)) == 4
    @test 16 <= tc <= log2(exp2(10)+exp2(16)+exp2(15)+exp2(9)+1e-8)
    @test sc == 11
    eincode3 = ein"(ab,acd),bcef,e,df->"
    Random.seed!(2)
    optcode3 = optimize_code(eincode3, size_dict, GreedyMethod()) 
    tc, sc = contraction_complexity(optcode3, edge_sizes)
    @test 16 <= tc <= log2(exp2(10)+exp2(16)+exp2(15)+exp2(9)+1e-8)

    optcode4 = optimize_code(eincode3, size_dict, ExactTreewidth()) 
    tc, sc = contraction_complexity(optcode4, edge_sizes)
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
    tc, sc = contraction_complexity(code, edge_sizes)
    @test tc == 60
    @test sc == 0
    optcode = optimize_code(code, size_dict, TreeSA(ntrials=1); simplifier=MergeVectors())
    tc2, sc2 = contraction_complexity(optcode, edge_sizes)
    @test sc2 == 10
    xs = vcat([TropicalF64.([-1 1; 1 -1]) for i=1:90], [TropicalF64.([0, 0]) for i=1:60])
    @test OMEinsum.flatten(optcode) == code
    @test OMEinsum.flatten(code) == code
    @test optcode(xs...)[].n == 66

    # slicer
    slicer = TreeSASlicer(score=ScoreFunction(sc_target=8))
    scode = slice_code(optcode, size_dict, slicer)
    @test scode isa SlicedEinsum
    @test contraction_complexity(scode, edge_sizes).sc == 8
    @test scode(xs...)[].n == 66
end

@testset "regression test" begin
    code = ein"i->"
    optcode = optimize_code(code, Dict('i'=>3), GreedyMethod())
    @test optcode isa NestedEinsum
    x = randn(3)
    @test optcode(x) ≈ code(x)

    code = ein"i,j->"
    optcode = optimize_code(code, Dict('i'=>3, 'j'=>3), GreedyMethod())
    @test optcode isa NestedEinsum
    x = randn(3)
    y = randn(3)
    @test optcode(x, y) ≈ code(x, y)

    code = ein"ij,jk,kl->ijl"
    optcode = optimize_code(code, Dict('i'=>3, 'j'=>3, 'k'=>3, 'l'=>3), GreedyMethod())
    @test optcode isa NestedEinsum
    a, b, c = [rand(3,3) for i=1:3]
    @test optcode(a, b, c) ≈ code(a, b, c)
end

@testset "simplifier and permute optimizer" begin
    code = EinCode([['a','b'], ['b','c'], ['c','d']], ['a','d'])
    code = optimize_code(code, uniformsize(code, 2), GreedyMethod())
    xs = [randn(3,3) for i=1:4]
    c2 = optimize_permute(code)
    @test code(xs...) ≈ c2(xs...)
end

@testset "save load" begin
    for code in [
        EinCode([[1,2], [2,3], [3,4]], [1,4]),
        EinCode([['a','b'], ['b','c'], ['c','d']], ['a','d'])
    ]
        gcode = optimize_code(code, uniformsize(code, 2), GreedyMethod())
        scode = optimize_code(code, uniformsize(code, 2), GreedyMethod(); slicer=TreeSASlicer(ntrials=1))
        for optcode in [gcode, scode]
            filename = tempname()
            writejson(filename, optcode)
            code2 = readjson(filename)
            @test optcode == code2
        end
    end
end

using LuxorGraphPlot

@testset "visualization tool" begin
    eincode = ein"ab,acd,bcef,e,df->"
    nested_ein = optein"ab,acd,bcef,e,df->"

    graph_1 = viz_eins(eincode)
    @test graph_1 isa LuxorGraphPlot.Luxor.Drawing 

    graph_2 = viz_eins(nested_ein)
    @test graph_2 isa LuxorGraphPlot.Luxor.Drawing

    gif = viz_contraction(nested_ein, filename = tempname() * ".gif")
    @test gif isa String

    video = viz_contraction(nested_ein)
    @test video isa String
end