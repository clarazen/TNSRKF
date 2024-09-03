module functionsBasisfunctions

using LinearAlgebra
using SparseArrays
using ..functions_KhatriRao_Kronecker

export basisfunctionsSE!

function basisfunctionsSE!(M::Vector{Int},X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector{Float64},Φ_::Vector{Matrix{Float64}})
    # computes Φ_, such that Φ_*Φ_' approx K as Khatri Rao multipliers
        D = size(X,2)
        for d = 1:D
            w        = collect(1:M[d])';
            sqrtΛ    = sqrt.(σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 ))
            Φ_[d]    = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w).*sqrtΛ';
        end
    
        return Φ_
end

function basisfunctionsSE!(M::Vector{Int},X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector{Float64},Φ,Λ)
    # computes Φ (as Khatri Rao multipliers) and 𝝠 such that Φ*Λ*Φ' approx K_SE 
        D = size(X,2)
        for d = 1:D
            w           = collect(1:M[d])';
            Λ[d]        = σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 ) 
            Φ[d]        = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w);
        end
    
        return Φ,Λ
end 

function basisfunctionsSE!(M::Vector{Int},X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector{Float64},Φ_::Vector{Matrix})
    # computes Φ_, such that Φ_*Φ_' approx K as Khatri Rao multipliers
        D = size(X,2)
        for d = 1:D
            w        = collect(1:M[d])';
            sqrtΛ    = sqrt.(σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 ))
            Φ_[d]    = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w).*sqrtΛ';
        end
    
        return Φ_
end

end

