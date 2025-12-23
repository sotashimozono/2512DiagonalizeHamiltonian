module BlockDMRG

using SparseArrays, LinearAlgebra, Arpack

A ⊗ B = kron(A, B)
const Id = sparse(I, 2, 2)
const σx = sparse([0.0 1.0; 1.0 0.0])
const σy = sparse([0.0 -im; im 0.0])
const σz = sparse([1.0 0.0; 0.0 -1.0])
export Id, σx, σy, σz, ⊗

abstract type AbstractBlockDMRG end
struct Block <: AbstractBlockDMRG
    dim::Int
    H::SparseMatrixCSC{ComplexF64,Int64}
    Op::Dict{Symbol,SparseMatrixCSC{ComplexF64,Int64}}
end
mutable struct FiniteDMRG <: AbstractBlockDMRG
    systemsize::Int
    unit_size::Int
    leftblocks::Vector{Block}
    rightblocks::Vector{Block}
    m_trunc::Int
end
mutable struct InfiniteDMRG <: AbstractBlockDMRG
    unit_length::Int
    leftblock::Block
    rightblock::Block
    m_trunc::Int
end
mutable struct SuperBlock <: AbstractBlockDMRG
    dim::Int
    H::SparseMatrixCSC{ComplexF64,Int64}
end

abstract type DMRGModel end
# Model interfaces
function GetLocalHamiltonian(model::DMRGModel, left::Block, right::Block)
    return GetLocalHamiltonian(model, left.Op, right.Op)
end
function GetLocalHamiltonian(model::DMRGModel, Op1, Op2)
    return error("Not implemented for generic DMRGModel")
end
function InitialBlock(model::DMRGModel)
    dim = 2
    H = spzeros(ComplexF64, dim, dim)
    block = Block(dim, H, Dict{Symbol,SparseMatrixCSC{ComplexF64,Int64}}())
    for (name, op) in model.Op
        block.Op[name] = op
    end
    return block
end
# example : Transverse Field Ising Model
@kwdef struct TFIM <: DMRGModel
    J::Float64 = 1.0
    h::Float64 = 0.5
    Op::Dict{Symbol,SparseMatrixCSC{ComplexF64,Int64}} = Dict(
        :I => Id, :Sx => σx, :Sz => σz
    )
end
export TFIM
function GetLocalHamiltonian(model::TFIM, Op1, Op2)
    J, h = model.J, model.h / 2
    H_local = -J * Op1[:Sz] ⊗ Op2[:Sz] - h * (Op1[:Sx] ⊗ Op2[:I] + Op1[:I] ⊗ Op2[:Sx])
    return Hermitian(sparse(H_local))
end
function GetLocalHamiltonian(model::TFIM, left::Block, right::Block)
    return GetLocalHamiltonian(model, left.Op, right.Op)
end

function EnlargeBlock(model::DMRGModel, block1::Block, block2::Block; from::Symbol=:left)
    if from == :left
        return EnlargeBlockFromLeft(model, block1, block2)
    elseif from == :right
        return EnlargeBlockFromRight(model, block1, block2)
    else
        error("Invalid 'from' argument. Use :left or :right.")
    end
end
# Enlarge Block from left side
function EnlargeBlockFromLeft(model::DMRGModel, block1::Block, block2::Block)
    dim_enlarged = block1.dim * block2.dim
    H_enlarged = spzeros(ComplexF64, dim_enlarged, dim_enlarged)

    H_environment = block1.H ⊗ block2.Op[:I]
    H_interact = GetLocalHamiltonian(model, block1, block2)
    H_enlarged = H_environment + H_interact
    H_enlarged = sparse(H_enlarged)

    Op_enlarged = Dict{Symbol,SparseMatrixCSC{ComplexF64,Int64}}()
    for (key, op) in block2.Op
        Op_enlarged[key] = block1.Op[:I] ⊗ op
    end

    return Block(dim_enlarged, H_enlarged, Op_enlarged)
end
# Enlarge Block from right side

function EnlargeBlockFromRight(model::DMRGModel, block1::Block, block2::Block)
    dim_enlarged = block1.dim * block2.dim
    H_enlarged = spzeros(ComplexF64, dim_enlarged, dim_enlarged)

    H_environment = block1.Op[:I] ⊗ block2.H
    H_interact = GetLocalHamiltonian(model, block1, block2)
    H_enlarged = H_environment + H_interact
    H_enlarged = sparse(H_enlarged)

    Op_enlarged = Dict{Symbol,SparseMatrixCSC{ComplexF64,Int64}}()
    for (key, op) in block1.Op
        Op_enlarged[key] = op ⊗ block2.Op[:I]
    end

    return Block(dim_enlarged, H_enlarged, Op_enlarged)
end
function CloseBlock(model::DMRGModel, block1::Block, block2::Block)
    dim_enlarged = block1.dim * block2.dim
    H_enlarged = spzeros(ComplexF64, dim_enlarged, dim_enlarged)

    H_env_left = block1.Op[:I] ⊗ block2.H
    H_env_right = block1.H ⊗ block2.Op[:I]
    H_environment = H_env_left + H_env_right
    H_interact = GetLocalHamiltonian(model, block1, block2)
    H_enlarged = H_environment + H_interact
    H_enlarged = sparse(H_enlarged)

    Op_enlarged = Dict{Symbol,SparseMatrixCSC{ComplexF64,Int64}}()
    return Block(dim_enlarged, H_enlarged, Op_enlarged)
end

function GetSuperBlock(model::DMRGModel, idmrg::InfiniteDMRG)
    leftblock = idmrg.leftblock
    siteblock = InitialBlock(model)
    unit_size = idmrg.unit_length
    rightblock = idmrg.rightblock
    # todo : construct superblock hamiltonian
    superblock = EnlargeBlockFromLeft(model, leftblock, siteblock)
    if unit_size > 1
        for i in 2:unit_size
            superblock = EnlargeBlockFromLeft(model, superblock, siteblock)
        end
    end
    superblock = CloseBlock(model, superblock, rightblock)
    return superblock
end

init_block = InitialBlock(TFIM())
m_trunc = 10
d = 2
unit_length = 10
idmrg = InfiniteDMRG(unit_length, init_block, init_block, m_trunc)
superblock = GetSuperBlock(TFIM(), idmrg)

# :S smallest, :L largest; :R real part, :I imaginary part
values, vectors = eigs(superblock.H; which=:SR, nev=10)
evals = real(values[1])
evecs = vectors[:, 1]

left = 2^div(unit_length, 2)
right = 2^(unit_length - div(unit_length, 2))
ρ = reshape(evecs, (idmrg.leftblock.dim * left, right * idmrg.rightblock.dim))

ρ_left = ρ * ρ'
U, S, V = svd(ρ_left)
mm = min(m_trunc, size(S, 1))

U_trunc = U[:, 1:mm]
S_trunc = S[1:mm]

@show size(U_trunc)
@show size(superblock.H)
@show size(S_trunc)

function truncate_block(block::Block, U_trunc::Matrix{ComplexF64}, m_trunc::Int)
    H_trunc = U_trunc' * block.H * U_trunc
    Op_trunc = Dict{Symbol,SparseMatrixCSC{ComplexF64,Int64}}()
    for (key, op) in block.Op
        Op_trunc[key] = U_trunc' * op * U_trunc
    end
    return Block(m_trunc, H_trunc, Op_trunc)
end

# idmrg.leftblock = truncate_block(idmrg.leftblock, U_trunc, mm)
# idmrg.rightblock = truncate_block(idmrg.rightblock, U_trunc, mm)
function get_superblock(fdmrg::FiniteDMRG, model::DMRGModel) end

end