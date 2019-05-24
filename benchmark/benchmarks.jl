using BenchmarkTools
using Random
using TensorOperations

Random.seed!(0)

const SUITE = BenchmarkGroup()

# Matrix multiplication
SUITE["matmul"] = BenchmarkGroup()
suite = SUITE["matmul"]

matmul(m1,m2) = @tensor out[i,j] := m1[i,l] * m2[l,j]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => rand(T,2,2),
            "small"  => rand(T,10,10),
            "medium" => rand(T,10^2,10^2),
            "large"  => rand(T, 10^3,10^3)]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable matmul($m,$m)
    end
end

#inner - reduction to scalar
SUITE["dot"] = BenchmarkGroup()
suite = SUITE["dot"]

mydot(m,m2) = @tensor out = m[i,j,k] * m2[i,j,k]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => rand(T,fill(2,3)...)
            "small"  => rand(T,fill(10,3)...)
            "medium" => rand(T,fill(20,3)...)
            "large"  => rand(T,fill(50,3)...)
            "huge"   => rand(T,fill(10^2,3)...)]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable mydot($m, $m)
    end
end

#trace
SUITE["trace"] = BenchmarkGroup()
suite = SUITE["trace"]

mytr(m) = @tensor out = m[i,i]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => rand(T,fill(2,2)...)
            "small"  => rand(T,fill(10,2)...)
            "medium" => rand(T,fill(10^2,2)...)
            "large"  => rand(T,fill(10^3,2)...)]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable mytr($m)
    end
end

#partial trace
SUITE["ptrace"] = BenchmarkGroup()
suite = SUITE["ptrace"]

ptrace(m) = @tensor out[i] := m[j,j,i]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args = ["tiny"   => rand(T,fill(2,3)...)
            "small"  => rand(T,fill(10,3)...)
            "medium" => rand(T,fill(20,3)...)
            "large"  => rand(T,fill(50,3)...)
            "huge"   => rand(T,fill(10^2,3)...)]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable ptrace($m)
    end
end

#permutation
SUITE["perm"] = BenchmarkGroup()
suite = SUITE["perm"]

myperm(m) = @tensor out[l,j,k,i] := m[i,j,k,l]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
            "tiny"   => rand(T,fill(1,4)...)
            "small"  => rand(T,fill(2,4)...)
            "medium" => rand(T,fill(10,4)...)
            "large"  => rand(T,fill(30,4)...)
            ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable myperm($m)
    end
end

# tensor contraction
SUITE["tcontract"] = BenchmarkGroup()
suite = SUITE["tcontract"]

tcontract(m1,m2) = @tensor out[i,j] := m1[i,k,l] * m2[k,j,l]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
            "tiny"   => rand(T,fill(1,3)...)
            "small"  => rand(T,fill(2,3)...)
            "medium" => rand(T,fill(10,3)...)
            "large"  => rand(T,fill(30,3)...)
            ]

    for (k,m) in args
        suite[string(T)][k] = @benchmarkable tcontract($m,$m)
    end
end

# Outer
SUITE["outer"] = BenchmarkGroup()
suite = SUITE["outer"]

myouter(m1,m2) = @tensor out[i,j,k,l] := m1[i,j] * m2[k,l]
for T in (Float32, Float64, ComplexF32, ComplexF64)
    suite[string(T)] = BenchmarkGroup()
    args =  [
            "tiny"   => rand(T,fill(2,2)...)
            "small"  => rand(T,fill(10,2)...)
            "medium" => rand(T,fill(50,2)...)
            "large"  => rand(T,fill(100,2)...)
            ]
    for (k,m) in args
        suite[string(T)][k] = @benchmarkable myouter($m, $m)
    end
end
