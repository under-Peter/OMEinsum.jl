using BenchmarkTools
using Random
using OMEinsum, CuArray

using CUDAnative: device!
error()
device!(0)
CuArrays.allowscalar(false)

Random.seed!(0)

const SUITE = BenchmarkGroup()

# Matrix multiplication
SUITE["matmul"] = BenchmarkGroup()
suite = SUITE["matmul"]

function matmul(a,b)
    CuArrays.@sync ein"ij,jk -> ik"(a,b)
end
￼
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => CuArray(rand(T,2,2)),
    "small"  => CuArray(rand(T,10,10)),
    "medium" => CuArray(rand(T,10^2,10^2)),
    "large"  => CuArray(rand(T, 10^3,10^3))
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable matmul($m,$m)
    end
end

# Matrix batch-multiplication
SUITE["batchmul"] = BenchmarkGroup()
suite = SUITE["batchmul"]


function batchmul(a,b)
    CuArrays.@sync ein"ijk,jlk -> ilk"(a,b)
end

for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => CuArray(rand(T,2,2,3)),
    "small"  => CuArray(rand(T,10,10,3)),
    "medium" => CuArray(rand(T,10^2,10^2,3)),
    "large"  => CuArray(rand(T, 10^3,10^3,3))
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable batchmul($m,$m)
    end
end

function mydot(a,b)
    CuArrays.@sync ein"ijk,ijk -> "(a,b)
end
#inner - reduction to scalar
SUITE["dot"] = BenchmarkGroup()
suite = SUITE["dot"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => CuArray(rand(T,fill(2,3)...))
    "small"  => CuArray(rand(T,fill(10,3)...))
    "medium" => CuArray(rand(T,fill(20,3)...))
    "large"  => CuArray(rand(T,fill(50,3)...))
    "huge"   => CuArray(rand(T,fill(10^2,3)...))]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable mydot($m, $m)
    end
end

function mytrace(a)
    CuArrays.@sync ein"ii -> "(a)
end
#trace
SUITE["trace"] = BenchmarkGroup()
suite = SUITE["trace"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => CuArray(rand(T,fill(2,2)...))
    "small"  => CuArray(rand(T,fill(10,2)...))
    "medium" => CuArray(rand(T,fill(10^2,2)...))
    "large"  => CuArray(rand(T,fill(10^3,2)...))]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable mytrace($m)
    end
end

function myptrace(a)
    CuArrays.@sync ein"iij -> j"(a)
end
#partial trace
SUITE["ptrace"] = BenchmarkGroup()
suite = SUITE["ptrace"]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => CuArray(rand(T,fill(2,3)...))
    "small"  => CuArray(rand(T,fill(10,3)...))
    "medium" => CuArray(rand(T,fill(20,3)...))
    "large"  => CuArray(rand(T,fill(50,3)...))
    "huge"   => CuArray(rand(T,fill(10^2,3)...))]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable myptrace($m)
    end
end

#diagonal
SUITE["diag"] = BenchmarkGroup()
suite = SUITE["diag"]

function mydiag(a)
    CuArrays.@sync ein"iij -> ij"(a)
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => CuArray(rand(T,fill(2,3)...))
    "small"  => CuArray(rand(T,fill(10,3)...))
    "medium" => CuArray(rand(T,fill(20,3)...))
    "large" => CuArray(rand(T,fill(30,3)...))
    "huge" => CuArray(rand(T,fill(100,3)...))
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable mydiag($m)
    end
end

#permutation
SUITE["perm"] = BenchmarkGroup()
suite = SUITE["perm"]

function myperm(a)
    CuArrays.@sync ein"ijkl -> ljki"(a)
end

for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => CuArray(rand(T,fill(1,4)...))
    "small"  => CuArray(rand(T,fill(2,4)...))
    "medium" => CuArray(rand(T,fill(10,4)...))
    "large"  => CuArray(rand(T,fill(30,4)...))
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable myperm($m)
    end
end

# tensor contraction
SUITE["tcontract"] = BenchmarkGroup()
suite = SUITE["tcontract"]

function tcontract(a,b)
    CuArrays.@sync ein"ijk, jlk -> il"(a,b)
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => CuArray(rand(T,fill(1,3)...))
    "small"  => CuArray(rand(T,fill(2,3)...))
    "medium" => CuArray(rand(T,fill(10,3)...))
    "large"  => CuArray(rand(T,fill(30,3)...))
    ]

    for (k,m) in args
        suite[string(T)][k] = @benchmarkable tcontract($m,$m)
    end
end

# star contraction
SUITE["star"] = BenchmarkGroup()
suite = SUITE["star"]

function mystar(a,b,c)
    CuArrays.@sync ein"ij,ik,il -> jkl"(a,b,c)
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => CuArray(rand(T,fill(2,2)...))
    "small"  => CuArray(rand(T,fill(10,2)...))
    "medium" => CuArray(rand(T,fill(30,2)...))
    "large"  => CuArray(rand(T,fill(100,2)...))
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable mystar($m,$m,$m)
    end
end

#star and contract
SUITE["starandcontract"] = BenchmarkGroup()
suite = SUITE["starandcontract"]

function mystarandcontract(a,b,c)
    CuArrays.@sync ein"ij,ik,ik -> j"(a,b,c)
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => CuArray(rand(T,fill(2,2)...))
    "small"  => CuArray(rand(T,fill(10,2)...))
    "medium" => CuArray(rand(T,fill(30,2)...))
    "large"  => CuArray(rand(T,fill(100,2)...))
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable mystarandcontract($m,$m,$m)
    end
end

# index-sum
SUITE["indexsum"] = BenchmarkGroup()
suite = SUITE["indexsum"]

function indexsum(a)
    CuArrays.@sync ein"ijk -> ik"(a)
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => CuArray(rand(T,fill(2,3)...))
    "small"  => CuArray(rand(T,fill(10,3)...))
    "medium" => CuArray(rand(T,fill(30,3)...))
    "large"  => CuArray(rand(T,fill(100,3)...))
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable indexsum($m)
    end
end

# Hadamard
SUITE["hadamard"] = BenchmarkGroup()
suite = SUITE["hadamard"]

function hadamard(a,b)
    CuArrays.@sync ein"ijk,ijk -> ijk"(a,b)
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => CuArray(rand(T,fill(2,3)...))
    "small"  => CuArray(rand(T,fill(10,3)...))
    "medium" => CuArray(rand(T,fill(30,3)...))
    "large"  => CuArray(rand(T,fill(100,3)...))
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable hadamard($m,$m)
    end
end

# Outer
SUITE["outer"] = BenchmarkGroup()
suite = SUITE["outer"]

function myouter(a,b)
    CuArrays.@sync ein"ij,kl -> ijkl"(a,b)
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
    "tiny"   => CuArray(rand(T,fill(2,2)...))
    "small"  => CuArray(rand(T,fill(10,2)...))
    "medium" => CuArray(rand(T,fill(50,2)...))
    "large"  => CuArray(rand(T,fill(100,2)...))
    ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable myouter($m,$m)
    end
end
