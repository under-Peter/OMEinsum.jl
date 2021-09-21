using BenchmarkTools
using Random
using OMEinsum


Random.seed!(0)

const SUITE = BenchmarkGroup()

SUITE["manyinds"] = BenchmarkGroup()
suite = SUITE["manyinds"]

for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["abcdefghijklmnop,flnqrcipstujvgamdwxyz->bcdeghkmnopqrstuvwxyz"]
    dims = [2]
    for str in args, dim in dims
        code = ein"abcdefghijklmnop,flnqrcipstujvgamdwxyz->bcdeghkmnopqrstuvwxyz"
        arr1 = rand(T, map(i->2, OMEinsum.getixs(code)[1])...)
        arr2 = rand(T, map(i->2, OMEinsum.getixs(code)[2])...)
        suite[string(T)]["tiny"] = @benchmarkable code($arr1,$arr2)
    end
end

# Matrix multiplication
SUITE["matmul"] = BenchmarkGroup()
suite = SUITE["matmul"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => rand(T,2,2),
    "small"  => rand(T,10,10),
    "medium" => rand(T,10^2,10^2),
    "large"  => rand(T, 10^3,10^3)
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ij,jk -> ik"($m,$m)
    end
end

# Matrix batch-multiplication
SUITE["batchmul"] = BenchmarkGroup()
suite = SUITE["batchmul"]


for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => rand(T,2,2,3),
    "small"  => rand(T,10,10,3),
    "medium" => rand(T,10^2,10^2,3),
    "large"  => rand(T, 10^3,10^3,3)
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ijk,jlk -> ilk"($m,$m)
    end
end

#inner - reduction to scalar
SUITE["dot"] = BenchmarkGroup()
suite = SUITE["dot"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => rand(T,fill(2,3)...)
    "small"  => rand(T,fill(10,3)...)
    "medium" => rand(T,fill(20,3)...)
    "large"  => rand(T,fill(50,3)...)
    "huge"   => rand(T,fill(10^2,3)...)]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ijk,ijk -> "($m, $m)
    end
end

#trace
SUITE["trace"] = BenchmarkGroup()
suite = SUITE["trace"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => rand(T,fill(2,2)...)
    "small"  => rand(T,fill(10,2)...)
    "medium" => rand(T,fill(10^2,2)...)
    "large"  => rand(T,fill(10^3,2)...)]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ii -> "($m)
    end
end

#partial trace
SUITE["ptrace"] = BenchmarkGroup()
suite = SUITE["ptrace"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => rand(T,fill(2,3)...)
    "small"  => rand(T,fill(10,3)...)
    "medium" => rand(T,fill(20,3)...)
    "large"  => rand(T,fill(50,3)...)
    "huge"   => rand(T,fill(10^2,3)...)]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"iij -> j"($m)
    end
end

#diagonal
SUITE["diag"] = BenchmarkGroup()
suite = SUITE["diag"]

for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => rand(T,fill(2,3)...)
    "small"  => rand(T,fill(10,3)...)
    "medium" => rand(T,fill(20,3)...)
    "large" => rand(T,fill(30,3)...)
    "huge" => rand(T,fill(100,3)...)
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ijj -> ij"($m)
    end
end

#permutation
SUITE["perm"] = BenchmarkGroup()
suite = SUITE["perm"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => rand(T,fill(1,4)...)
    "small"  => rand(T,fill(2,4)...)
    "medium" => rand(T,fill(10,4)...)
    "large"  => rand(T,fill(30,4)...)
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ijkl -> ljki"($m)
    end
end

# tensor contraction
SUITE["tcontract"] = BenchmarkGroup()
suite = SUITE["tcontract"]

for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => rand(T,fill(1,3)...)
    "small"  => rand(T,fill(2,3)...)
    "medium" => rand(T,fill(10,3)...)
    "large"  => rand(T,fill(30,3)...)
    ]

    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ijk, jlk -> il"($m,$m)
    end
end

# star contraction
SUITE["star"] = BenchmarkGroup()
suite = SUITE["star"]

for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => rand(T,fill(2,2)...)
    "small"  => rand(T,fill(10,2)...)
    "medium" => rand(T,fill(30,2)...)
    "large"  => rand(T,fill(100,2)...)
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ij,ik,il -> jkl"($m,$m,$m)
    end
end

#star and contract
SUITE["starandcontract"] = BenchmarkGroup()
suite = SUITE["starandcontract"]

for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => rand(T,fill(2,2)...)
    "small"  => rand(T,fill(10,2)...)
    "medium" => rand(T,fill(30,2)...)
    "large"  => rand(T,fill(100,2)...)
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ij,ik,ik -> j"($m,$m,$m)
    end
end

# index-sum
SUITE["indexsum"] = BenchmarkGroup()
suite = SUITE["indexsum"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => rand(T,fill(2,3)...)
    "small"  => rand(T,fill(10,3)...)
    "medium" => rand(T,fill(30,3)...)
    "large"  => rand(T,fill(100,3)...)
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ijk -> ik"($m)
    end
end
# Hadamard
SUITE["hadamard"] = BenchmarkGroup()
suite = SUITE["hadamard"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => rand(T,fill(2,3)...)
    "small"  => rand(T,fill(10,3)...)
    "medium" => rand(T,fill(30,3)...)
    "large"  => rand(T,fill(100,3)...)
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ijk,ijk -> ijk"($m,$m)
    end
end

# Outer
SUITE["outer"] = BenchmarkGroup()
suite = SUITE["outer"]

for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => rand(T,fill(2,2)...)
    "small"  => rand(T,fill(10,2)...)
    "medium" => rand(T,fill(50,2)...)
    "large"  => rand(T,fill(100,2)...)
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ein"ij,kl -> ijkl"($m,$m)
    end
end
