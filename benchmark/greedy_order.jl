using OMEinsum, LightGraphs, BenchmarkTools

function uniformsize(@nospecialize(code::EinCode{ixs,iy}), size::Int) where {ixs, iy}
    Dict([c=>size for c in [OMEinsum.flatten(ixs)..., iy...]])
end
uniformsize(ne::OMEinsum.NestedEinsum, size::Int) = uniformsize(OMEinsum.flatten(ne), size)

function random_regular_eincode(n, k)
	g = LightGraphs.random_regular_graph(n, k)
	ixs = [minmax(e.src,e.dst) for e in LightGraphs.edges(g)]
	code = EinCode((ixs..., [(i,) for i in LightGraphs.vertices(g)]...), ())
end

number_of_nodes = 200
code = random_regular_eincode(number_of_nodes, 3)
optcode = @benchmark optimize_greedy($code, $(uniformsize(code, 2)); nrepeat=10, method=OMEinsum.MinSpaceOut())
