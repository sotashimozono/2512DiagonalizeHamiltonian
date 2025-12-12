module DiagonalizeHamiltonian

using Reexport
@reexport using SparseArrays, LinearAlgebra, Arpack
@reexport using Plots, Statistics, Random

const I = sparse([1.0 0.0; 0.0 1.0])
const σx = sparse([0.0 1.0; 1.0 0.0])
const σy = sparse([0.0 -im; im 0.0])
const σz = sparse([1.0 0.0; 0.0 -1.0])
export σx, σy, σz

const ↑ = [1.0; 0.0]
const ↓ = [0.0; 1.0]
export ↑, ↓

A ⊗ B = kron(A, B)
δE(λ) = λ[2:end] - λ[1:(end - 1)]
export ⊗, δE

#TODO : Distribution
poisson(E) = exp(-E)
export poisson
wigner_surmise(E) = (π / 2) * E * exp(-π * E^2 / 4)
export wigner_surmise

#TODO : Implement some models
abstract type AbstractHamiltonian end

abstract type BoundaryCondition <: AbstractHamiltonian end
struct PBC <: BoundaryCondition end
struct OBC <: BoundaryCondition end
struct SSD <: BoundaryCondition end
export PBC, OBC, SSD

abstract type QuantumSpinModel <: AbstractHamiltonian end
@kwdef struct TFIM <: QuantumSpinModel
  J::Float64 = 1.0
  h::Float64 = 0.50
  boundary::BoundaryCondition = PBC()
end
export TFIM
function (model::TFIM)(N::Int)
  H = spzeros(ComplexF64, 2^N, 2^N)
  matarray = [I for _ in 1:N]
  matarray_field = [I for _ in 1:N]
  matarray_ising = [I for _ in 1:N]
  for i in 1:N
    matfield = copy(matarray_field)
    matfield[i] = σx
    H += model.h * reduce(⊗, matfield)
    if model.boundary isa OBC && i == N
      continue
    end
    j = i % N + 1
    matising = copy(matarray_ising)
    matising[i] = σz
    matising[j] = σz
    H += model.J * reduce(⊗, matising)
  end
  return Hermitian(H)
end

@kwdef struct TFIML <: QuantumSpinModel
  J::Float64 = 1.0
  h::Float64 = 1.0
  g::Float64 = 0.10
  boundary::BoundaryCondition = PBC()
end
export TFIML
function (model::TFIML)(N::Int)
  H = spzeros(ComplexF64, 2^N, 2^N)

  matarray = [I for _ in 1:N]
  matarray_field = [I for _ in 1:N]
  matarray_ising = [I for _ in 1:N]
  for i in 1:N
    matfield = copy(matarray_field)
    matfield[i] = σx
    H += model.h * reduce(⊗, matfield)
    matfield[i] = σz
    H += model.g * reduce(⊗, matfield)
    if model.boundary isa OBC && i == N
      continue
    end
    j = i % N + 1
    matising = copy(matarray_ising)
    matising[i] = σz
    matising[j] = σz
    H += model.J * reduce(⊗, matising)
  end
  return Hermitian(H)
end

@kwdef struct XYZ <: QuantumSpinModel
  Jx::Float64 = 1.0
  Jy::Float64 = 1.0
  Jz::Float64 = 1.0
  hx::Float64 = 0.0
  hy::Float64 = 0.0
  hz::Float64 = 0.0
  boundary::BoundaryCondition = PBC()
end
export XYZ
function (model::XYZ)(N::Int)
  H = spzeros(ComplexF64, 2^N, 2^N)

  matarray = [I for _ in 1:N]
  for i in 1:N
    if model.boundary isa OBC && i == N
      continue
    end
    j = i % N + 1
    # Interaction terms
    matisingx = copy(matarray)
    matisingy = copy(matarray)
    matisingz = copy(matarray)
    matisingx[i] = σx
    matisingx[j] = σx
    matisingy[i] = σy
    matisingy[j] = σy
    matisingz[i] = σz
    matisingz[j] = σz
    H += model.Jx * reduce(⊗, matisingx)
    H += model.Jy * reduce(⊗, matisingy)
    H += model.Jz * reduce(⊗, matisingz)
    # Field terms
    matfieldx = copy(matarray)
    matfieldy = copy(matarray)
    matfieldz = copy(matarray)
    matfieldx[i] = σx
    matfieldy[i] = σy
    matfieldz[i] = σz
    H += model.hx * reduce(⊗, matfieldx)
    H += model.hy * reduce(⊗, matfieldy)
    H += model.hz * reduce(⊗, matfieldz)
  end
  return Hermitian(H)
end

end