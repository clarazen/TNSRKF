module functionsTTmatmul

using LinearAlgebra
using ..functionsTT
using ..functions_KhatriRao_Kronecker

export getttm, tt2ttm, khrtimesttm

function getttm(tt::TTv,d::Int)
    # computes the matrix W_{∖setminus d} in TT matrix format
    D           = order(tt)
    middlesizes = size(tt)
    M           = middlesizes[d]
    newms       = zeros(2,D)
    newms[1,:]  = middlesizes
    newms[2,:]  = ones(D)
    Wd          = tt2ttm(tt,Int.(newms))
    Wd[d]       = reshape(Matrix(I,(M,M)),(1,M,M,1))
    if d>1
        Wd[d-1] = permutedims(Wd[d-1],(1,2,4,3))
    end
    if d<D
        Wd[d+1] = permutedims(Wd[d+1],(3,2,1,4))
    end
    return Wd
end

function tt2ttm(tt::TTv,middlesizes::Matrix)
    cores = Vector{Array{Float64,4}}(undef,order(tt));
    rnks  = rank(tt);
    for i = 1:order(tt)
        cores[i] = reshape(tt[i],(rnks[i], middlesizes[1,i], middlesizes[2,i], rnks[i+1]));
    end
    TT(cores,tt.normcore)
end


function khrtimesttm(khr,ttm::TTm)
    # computes the produt of two matrices, where the first one (khr) has a 
    # row-wise Khatri-Rao structure and the second (ttm) is a TTm. 

    D       = order(ttm)
    N       = size(khr[1],1)
    M1      = size(ttm[1],2)
    M2      = size(ttm[2],2)
    O1      = size(ttm[1],3)
    O2      = size(ttm[2],3)
    R2      = size(ttm[1],4)
    R3      = size(ttm[2],4)

    tmp     = khr[1]*reshape(ttm[1], (M1,O1*R2) )
    tmp     = KhatriRao(tmp,Matrix(khr[2]),1) 
    
    Tmp     = permutedims(reshape(tmp, (N,M2,O1,R2) ), [1,3,4,2] )
    tmp     = reshape(Tmp, (N*O1,R2*M2) )
    tmp     = reshape(tmp*reshape(ttm[2], (R2*M2,O2*R3) ), (N,O1*O2*R3) )
    colsz   = (O1,O2)

    for d = 2:D-1     
        Mdd     = size(ttm[d+1],2)
        Odd     = size(ttm[d+1],3)
        Rdd     = size(ttm[d],4)
        Rddd    = size(ttm[d+1],4)
        
        tmp     = KhatriRao(tmp,Matrix(khr[d+1]),1)
        Tmp     = permutedims(reshape(tmp, (N,Mdd,colsz...,Rdd) ), [1,collect(3:d+2)...,d+3,2] )
        tmp     = reshape(Tmp, (N*prod(colsz),Mdd*Rdd) )
        tmp     = tmp * reshape(ttm[d+1],Rdd*Mdd,Odd*Rddd)
        tmp     = reshape(tmp, N,Int(length(tmp)/N) ) 

        colsz   = (colsz...,Odd)
        
    end
    return reshape(tmp, (N,prod(colsz)) )
end



end 