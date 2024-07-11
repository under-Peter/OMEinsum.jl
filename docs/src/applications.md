# Application

## List of packages using OMEinsum
- [GenericTensorNetworks](https://github.com/QuEraComputing/GenericTensorNetworks.jl), solving combinational optimization problems by generic tensor networks.
- [TensorInference](https://github.com/TensorBFS/TensorInference.jl), probabilistic inference using contraction of tensor networks
- [YaoToEinsum](https://github.com/QuantumBFS/Yao.jl), the tensor network simulation backend for quantum circuits.
- [TensorNetworkAD2](https://github.com/YidaiZhang/TensorNetworkAD2.jl), using differential programming tensor networks to solve quantum many-body problems.
- [TensorQEC](https://github.com/nzy1997/TensorQEC.jl), tensor networks for quantum error correction.

## Example: Solving a 3-coloring problem on the Petersen graph
Let us focus on graphs
with vertices with three edges each. A question one might ask is:
How many different ways are there to colour the edges of the graph with
three different colours such that no vertex has a duplicate colour on its edges?

The counting problem can be transformed into a contraction of rank-3 tensors
representing the edges. Consider the tensor `s` defined as
```@repl coloring
using OMEinsum
s = map(x->Int(length(unique(x.I)) == 3), CartesianIndices((3,3,3)))
```

Then we can simply contract `s` tensors to get the number of 3 colourings satisfying the above condition!
E.g. for two vertices, we get 6 distinct colourings:
```@repl coloring
ein"ijk,ijk->"(s,s)[]
```

Using that method, it's easy to find that e.g. the peterson graph allows no 3 colouring, since
```@repl coloring
code = ein"afl,bhn,cjf,dlh,enj,ago,big,cki,dmk,eom->"
afl, bhn, cjf, dlh, enj, ago, big, cki, dmk, eom 
code(fill(s, 10)...)[]
```

The peterson graph consists of 10 vertices and 15 edges and looks like a pentagram
embedded in a pentagon as depicted here:

![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Petersen_graph.svg/252px-Petersen_graph.svg.png)

`OMEinsum` does not optimie the contraction order by default, so the above contraction can be time consuming. To speed up the contraction, we can use `optimize_code` to optimize the contraction order:
```@repl coloring
optcode = optimize_code(code, uniformsize(code, 3), TreeSA())
contraction_complexity(optcode, uniformsize(optcode, 3))
optcode(fill(s, 10)...)[]
```
We can see the time complexity of the optimized code is much smaller than the original one. To know more about the contraction order optimization, please check the Julia package [`OMEinsumContractionOrders.jl`](https://github.com/TensorBFS/OMEinsumContractionOrders.jl).

Confronted with the above result, we can ask whether the peterson graph allows a relaxed variation of 3 colouring, having one vertex that might accept duplicate colours. The answer to that can be found using the gradient w.r.t a vertex:
```@repl coloring
using Zygote: gradient
gradient(x->optcode(x,s,s,s,s,s,s,s,s,s)[], s)[1] |> sum
```
This tells us that even if we allow duplicates on one vertex, there are no 3-colourings for the peterson graph.

