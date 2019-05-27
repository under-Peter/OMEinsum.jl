@testset "expanding" begin
    a = randn(20,10)
    b = zeros(20,10) # i -> ii
    expandall!(b, (1,2), a, (1,2))
    @test b ≈ a

    a = randn(20,10)
    b = zeros(10,20) # i -> ii
    expandall!(b, (2,1), a, (1,2))
    @test b ≈ permutedims(a)

    a = randn(10)
    b = zeros(10,10) # i -> ii
    expandall!(b, (1,1), a, (1,))
    @test b ≈ diagm(0=>a)

    a = reshape(rand(1))
    b = zeros(10,20) # -> ij
    expandall!(b,(1,2), a, ())
    @test b ≈ fill(a[1],10,20)

    a = reshape(rand(1))
    b = zeros(20,20) # -> ii
    expandall!(b ,(2,2), a, ())
    @test b ≈ diagm(0 => fill(a[1], 20))


    a = rand(10)
    b = zeros(20,20,10) # j -> iij
    ref = zeros(20,20,10)
    for (i,j) in Iterators.product(1:20,1:10)
        ref[i,i,j] = a[j]
    end

    expandall!(b, (2,2,1), a,(1,))
    @test ref ≈ b


    a = rand(10)
    b = zeros(20,10,20) # j -> iji
    ref = zeros(20,10,20)
    for (i,j) in Iterators.product(1:20,1:10)
        ref[i,j,i] = a[j]
    end
    expandall!(b, (2,1,2), a,(1,))
    @test ref ≈ b


    a = rand(10,30)
    b = zeros(30,20,10,20) # jk -> kiji
    ref = zeros(30,20,10,20)
    for (k,i,j) in Iterators.product(1:30,1:20,1:10)
        ref[k,i,j,i] = a[j,k]
    end
    expandall!(b, (2,3,1,3), a, (1,2))
    @test ref ≈ b
end
