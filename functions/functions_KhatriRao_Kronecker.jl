module functions_KhatriRao_Kronecker

using LinearAlgebra
using ..functionsTT

export khr2mat,KhatriRao,kr2ttm,khr2approxttm,factors2ttm,khr2ttm,getΦentries,repeatedkhr2ttm

function khr2mat(Φ::Vector{Matrix{Float64}})
    # computes the row-wise Khatri-Rao product for given set of matrices
    D     = size(Φ,1)
    Φ_mat = ones(size(Φ[D],1),1)
    for d = D:-1:1
        Φ_mat = KhatriRao(Φ_mat,Φ[d],1)
    end    
    return Φ_mat
end

function khr2mat(Φ::Vector{Vector{Float64}})
    # computes the row-wise Khatri-Rao product for given set of matrices
    D     = size(Φ,1)
    Φ_mat = ones(size(Φ[D],1),1)
    for d = D:-1:1
        Φ_mat = KhatriRao(Φ_mat,Φ[d],1)
    end    
    return Φ_mat
end

function kr2ttm(kr::Vector{Vector})
    D       = size(kr,2)
    cores   = Vector{Array}(undef,D)
    for d = 1:D
        Md       = size(kh[d],1)
        cores[d] = reshape(diagm(kr[d]),1,Md,Md,1)
    end
    return TT(cores)
end

function KhatriRao(A::Matrix{Float64},B::Matrix{Float64},dims::Int64)
    if dims == 1 # row-wise
        C = zeros(size(A,1),size(A,2)*size(B,2));
        @inbounds @simd for i = 1:size(A,1)
            @views kron!(C[i,:],A[i,:],B[i,:])
        end
    elseif dims == 2 # column-wise
        C = zeros(size(A,1)*size(B,1),size(A,2));
        @inbounds @simd for i = 1:size(A,2)
            @views kron!(C[:,i],A[:,i],B[:,i])
        end
    end

    return C
end

function khr2approxttm(Wdim)

    D               = size(Wdim,1)
    cores           = Vector{Array{Float64,4}}(undef,D)

    M               = size.(Wdim,2)
    N               = size(Wdim[1],1)
    sizes           = Int.(ones(D,4))
    sizes[1,:]      = [1,M[1],N,1]
    cores[1]        = reshape(Wdim[1]',Int.(Tuple(sizes[1,:])))
    
    for d in 1:D-1
        tmp         = reshape( cores[d], ( sizes[d,1]*sizes[d,2], sizes[d,3]*sizes[d,4] ) ) 
        tmp         = KhatriRao(Matrix(Wdim[d+1]'),tmp,2)
        tmp         = reshape(tmp,sizes[d,1]*M[d], M[d+1]*N)

        F           = svd(tmp)
        tol         = eps(F.S[1]) * maximum([N, M[d], M[d+1]])
        tr          = size(F.S)[1]#sum(F.S .> tol)

        sizes[d,2:end]      = [M[d], 1, tr]
        sizes[d+1,1:end-1]  = [tr, M[d+1],N]
        cores[d]    = reshape(F.U[:, 1:tr], Tuple(sizes[d,:]))
        cores[d+1]  = reshape(diagm(F.S[1:tr]) * F.Vt[1:tr,:], Tuple(sizes[d+1,:]) )
    end
   
    return TT(cores,D)
end

function factors2ttm(factors)
    D       = size(factors,1)
    cores   = Vector{Array{Float64,4}}(undef,D)
    for d = 1:D
        cores[d] = reshape(factors[d],1,size(factors[d],1),size(factors[d],2),1)
    end

    return TT(cores)
end

function khr2ttm(khr::Vector{Vector})
    return factors2ttm(khr)
end

function khr2ttm(khr::Vector{Matrix{Float64}})
    D   = size(khr,1)
    P   = size(khr[1],1)

    ttm   = Vector{TT}(undef,P)  
    for p = 1:P
        cores    = Vector{Array{Float64,4}}(undef,D)
        for d = 2:D
            cores[d] = reshape(khr[d][p,:],1,size(khr[d][p,:],1),1,1)
        end
        e01      = zeros(P)
        e01[p]   = 1
        cores[1] = reshape(kron(e01,khr[1][p,:]),1,size(khr[1][p,:],1),P,1)
        ttm[p]   = TT(cores)
    end

    ttsum = ttm[1]
    for p = 2:P
        ttsum      = ttsum + ttm[p]
    end

    return ttsum
end

function khr2ttm(khr::Vector{Matrix})
    D   = size(khr,1)
    P   = size(khr[1],1)

    ttm   = Vector{TT}(undef,P)  
    for p = 1:P
        cores    = Vector{Array{Float64,4}}(undef,D)
        for d = 2:D
            cores[d] = reshape(khr[d][p,:],1,size(khr[d][p,:],1),1,1)
        end
        e01      = zeros(P)
        e01[p]   = 1
        cores[1] = reshape(kron(e01,khr[1][p,:]),1,size(khr[1][p,:],1),P,1)
        ttm[p]   = TT(cores)
    end

    ttsum = ttm[1]
    for p = 2:P
        ttsum      = ttsum + ttm[p]
    end

    return ttsum
end

function repeatedkhr2ttm(Ũ::Vector{Matrix{Float64}},ϵ::Float64)
    # repeated Kathti-Rao product into tensor train matrix
    # Source: Batselier et al.: Tensor network subspace identification of polynomial state space models, Algorithm 2
    
    D    = length(Ũ);
    M    = size(Ũ[1],1);
    P    = size(Ũ[1],2);
    U    = Vector{Array{Float64,4}}(undef,D);
    U[1] = reshape(Ũ[D],(1,M,P,1));
    r1   = 1; err2 = 0;
    δ    = ϵ / sqrt(D-1) * norm(.*(Ũ));
    for i = 1:D-1
        T = reshape(U[i],(r1*M,P));
        Ttmp = KhatriRao(Ũ[D-i],T,2)

        T = reshape(Ttmp,(r1*M,M*P));
        F = svd!(T);
        r2 = length(F.S);
        sv2 = cumsum(reverse(F.S).^2);
        tr  = findfirst(sv2 .> δ^2);
        if typeof(tr) == Nothing
            tr = length(sv2)-1;
        else
            tr = Int(tr)-1;
        end
        if tr > 0
            r2 = r2 - tr;
            err2 += sv2[tr];
        end

        U[i] = reshape(F.U[:,1:r2],(r1,M,1,r2));
        U[i+1] = reshape(diagm(F.S[1:r2])*F.Vt[1:r2,:],(r2,M,P,1));
        r1 = r2;
    end
    return TT(U), err2
end

function getΦentries(Φ,ind::UnitRange{Int64})
    D = size(Φ,1)
    Φentries = Vector{Matrix}(undef,D)
    for d = 1:D
        Φentries[d] = Φ[d][ind,:]
    end

    return Φentries
end

function getΦentries(Φ,ind::Int)
    D = size(Φ,1)
    Φentries = Vector{Vector}(undef,D)
    for d = 1:D
        Φentries[d] = Φ[d][ind[1],:]
    end

    return Φentries
end

end