using DiagonalizeHamiltonian.BlockDMRG
using SparseArrays, LinearAlgebra

@testset "Definitions" begin
    @test Id[1, 1] == 1.0
    @test Id[2, 2] == 1.0
    @test Id[1, 2] == 0.0
    @test Id[2, 1] == 0.0
    @test σx[1, 2] == 1.0
    @test σx[2, 1] == 1.0
    @test σx[1, 1] == 0.0
    @test σx[2, 2] == 0.0
    @test σy[1, 2] == -im
    @test σy[2, 1] == im
    @test σy[1, 1] == 0.0
    @test σy[2, 2] == 0.0
    @test σz[1, 1] == 1.0
    @test σz[2, 2] == -1.0
    @test σz[1, 2] == 0.0
    @test σz[2, 1] == 0.0

    A = Id ⊗ Id
    @test A == sparse(I, 4, 4)
    B = σx ⊗ σz
    @test B[3, 1] == 1.0
    @test B[4, 2] == -1.0
    @test B[1, 3] == 1.0
    @test B[2, 4] == -1.0

    model = TFIM()
    @test model isa BlockDMRG.DMRGModel
    @test model.J == 1.0
    @test model.h == 0.5
    @test haskey(model.Op, :I)
    @test haskey(model.Op, :Sx)
    @test haskey(model.Op, :Sz)

    block = BlockDMRG.InitialBlock(model)
    @test block.dim == 2
    @test size(block.H) == (2, 2)
    @test block.H[1, 1] == 0.0
    @test block.Op[:I] == Id
    @test block.Op[:Sx] == σx
    @test block.Op[:Sz] == σz

    H = BlockDMRG.GetLocalHamiltonian(model, block, block)
    @test size(H) == (4, 4)
    @test issparse(H)

    block2 = BlockDMRG.EnlargeBlock(model, block, block; from=:left)
    @show block2
end