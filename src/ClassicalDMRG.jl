module BlockDMRG


using SparseArrays, LinearAlgebra, Arpack
using Plots, Statistics, Random

const Id = sparse(I, 2, 2)
const σx = sparse([0.0 1.0; 1.0 0.0])
const σy = sparse([0.0 -im; im 0.0])
const σz = sparse([1.0 0.0; 0.0 -1.0])
export σx, σy, σz

const ↑ = [1.0; 0.0]
const ↓ = [0.0; 1.0]
export ↑, ↓

abstract type BlockDMRG end
struct Block <: BlockDMRG
    dim::Int
    H::SparseMatrixCSC{ComplexF64, Int64}
    Op::Dict{String, SparseMatrixCSC{ComplexF64, Int64}}
end
struct FiniteDMRG <: BlockDMRG
    systemsize::Int
    unit_size::Int
    leftblocks::Vector{Block}
    rightblocks::Vector{Block}
    m_trunc::Int
end
struct InfiniteDMRG <: BlockDMRG
    unit_leght::Int
    leftblock::Block
    rightblock::Block
    m_trunc::Int
end

abstract type DMRGModel end
@kwdef struct TFIM <: DMRGModel
    J::Float64 = 1.0
    h::Float64 = 0.5
    Op::Dict{String, SparseMatrixCSC{ComplexF64, Int64}} = Dict(
        :I => Id,
        :Sx => σx,
        :Sz => σz,
    )
end
function local_hamiltonian(Op::Dict{String, SparseMatrixCSC{ComplexF64, Int64}}, Op::Dict{String, SparseMatrixCSC{ComplexF64, Int64}}, model::TFIM)
    J = model.J
    h = model.h

    H_loc = -J * kron(Op["Sz"], Op["Sz"]) - h * (kron(Op["Sx"], Op["I"]) + kron(Op["I"], Op["Sx"]))
    return sparse(H_loc)
end
struct Heisenberg <: DMRGModel end

function get_initial_block(model::DMRGModel)
    error("get_initial_block not implemented for $(typeof(model))")
end
function get_superblock(fdmrg::FiniteDMRG, model::DMRGModel)

end