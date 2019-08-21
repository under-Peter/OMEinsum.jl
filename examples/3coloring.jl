# 3-coloring problem
# https://arxiv.org/abs/1708.00006
using OMEinsum
using Zygote: gradient
s = map(x->Int(length(unique(x.I)) == 3), CartesianIndices((3,3,3)))
c = ein"afl,bhn,cjf,dlh,enj,ago,big,cki,dmk,eom->"(fill(s, 10)...)
println("number of possible coloring is $c")

# get relaxation of tensor
gradc, = gradient(x->ein"afl,bhn,cjf,dlh,enj,ago,big,cki,dmk,eom->"(x,s,s,s,s,s,s,s,s,s)[], s)
println("number of possible coloring after relaxing one vertex is $(sum(gradc))")
