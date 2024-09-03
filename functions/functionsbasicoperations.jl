module functionsbasicoperations

import Base: +,*,-
using ..functionsTT

export +,*,-,outerproduct,multiplyTT

    # summation of two TTm's
    function +(ttm1::TT{4},ttm2::TT{4})
        D        = order(ttm1);
        cores    = Vector{Array{Float64,4}}(undef,D);
        cores[1] = cat(ttm1[1], ttm2[1],dims=4);
        cores[D] = cat(ttm1[D], ttm2[D],dims=1);
        sizes = size(ttm1);
        rnks1 = rank(ttm1);
        rnks2 = rank(ttm2);
        for d = 2:D-1    
            row1     = cat(ttm1[d], zeros((rnks1[d],sizes[:,d]...,rnks2[d+1])),dims=4);
            row2     = cat(zeros((rnks2[d],sizes[:,d]...,rnks1[d+1])), ttm2[d],dims=4);
            cores[d] = cat(row1,row2,dims=1);
        end
                
        TT(cores);
    end

    function +(tt1::TT{3},tt2::TT{3})
        D        = order(tt1);
        cores    = Vector{Array{Float64,3}}(undef,D);
        cores[1] = cat(tt1[1], tt2[1],dims=3);
        cores[D] = cat(tt1[D], tt2[D],dims=1);
        sizes = size(tt1);
        rnks1 = rank(tt1);
        rnks2 = rank(tt2);
        for d = 2:D-1    
            row1     = cat(tt1[d], zeros((rnks1[d],sizes[d],rnks2[d+1])),dims=3);
            row2     = cat(zeros((rnks2[d],sizes[d],rnks1[d+1])), tt2[d],dims=3);
            cores[d] = cat(row1,row2,dims=1);
        end
                
        TT(cores);
    end

    function -(ttm1::TT,ttm2::TT)
        minusttm2 = copyTT(ttm2)
        minusttm2[1] = -minusttm2[1]
        return minusttm2 + ttm1
    end
   
    function *(tt1::TTm,tt2::TTm)
        # product of two thin TTms, returns matrix
        mat     = ones(1,1)
        sizes   = Int[]
        D       = order(tt1)
        for d = 1:D
            szl   = size(tt1[d])
            left  = permutedims(tt1[d],[1,2,4,3])
            left  = reshape(left,szl[1]*szl[2]*szl[4],szl[3])

            szr   = size(tt2[d])
            right = permutedims(tt2[d],[2,1,3,4])
            right = reshape(right,szr[2],szr[1]*szr[3]*szr[4])

            tmp   = reshape(left*right,szl[1],szl[2],szl[4],szr[1],szr[3],szr[4])
            tmp   = permutedims(tmp,[1,4,2,5,3,6])

            mat   = mat*reshape(tmp,szl[1]*szr[1],szl[2]*szr[3]*szl[4]*szr[4])
            mat   = reshape(mat,size(mat,1)*szl[2]*szr[3],szl[4]*szr[4])
            push!(sizes,szl[2],szr[3])
        end
        mat = reshape(mat,Tuple(sizes))
        mat = permutedims(mat,vcat(collect(1:2:2D-1),collect(2:2:2D)))
        mat = reshape(mat,prod(sizes[1:2:2D-1]),prod(sizes[2:2:2D]))

        return mat

    end

    function *(tt1::TTm,tt2::TTv)
        # product of tall TTm and TTv, returns matrix with one column
        return tt1*ttv2ttm(tt2,Int.(vcat(size(tt2)',ones(1,order(tt2)))))
    end

    function *(ttm::TT,kr::Vector{Matrix{Float64}})
        # product of TTm with Kronecker product of matrices, returns TTm of same size
        D       = order(ttm);
        cores   = Vector{Array{Float64,4}}(undef,D)
        for d = 1:D
            sz       = size(ttm[d])
            tmp      = permutedims(ttm[d],[1,2,4,3])
        
            tmp      = reshape(tmp,      (sz[1]*sz[2]*sz[4],sz[3]))
            tmp      = reshape(tmp*kr[d],(sz[1],sz[2],sz[4],sz[3]))
            cores[d] = permutedims(tmp,[1,2,4,3])
        end
        return TT(cores)
    end

    function *(ttm::TTm,kr::Vector{Vector})
        # product of TTm with Kronecker product of vectors
        D       = order(ttm);
        cores   = Vector{Array{Float64,3}}(undef,D)
        for d = 1:D
            sz       = size(ttm[d])
            tmp      = permutedims(ttm[d],[1,2,4,3])
        
            tmp      = reshape(tmp,      (sz[1]*sz[2]*sz[4],sz[3]))
            cores[d] = reshape(tmp*kr[d],(sz[1],sz[2],sz[4]))
        end
        return TT(cores)
    end

    function *(kr::Vector{Vector},ttm::TT)
        return *(transpose(ttm),kr)
    end

    function *(kr::Vector{Vector{Float64}},tt::TTv)
        # inner product of TT and Kronecker product of vectors, returns scalar
        D       = order(tt);
        res     = 1
        for d = 1:D
            sz       = size(tt[d])
            tmp      = permutedims(tt[d],[2,1,3])
        
            tmp      = reshape(tmp,      (sz[2], sz[1]*sz[3]))
            tmp      = reshape(kr[d]'*tmp,(sz[1],sz[3]))
            res      = res*tmp
        end
        return res[1]
    end

    function *(ttm::TTm,mat::Matrix) # check indices
        ttm_    = copyTT(ttm)
        ttm_[1] = permutedims(tmult(ttm[1],mat,[[1,2,4],[3]],[[1],[2]]),[1,2,4,3])
        return ttm_
    end

    
    function multiplyTT(tt1::TTm,tt2::TTm)
        D       = order(tt1)
        cores   = Vector{Array{Float64,4}}(undef,D)
        for d = 1:D
            tmp         = tmult(tt1[d],tt2[d],[[1,2,4],[3]],[[2],[1,3,4]])
            sz          = size(tmp)
            tmp         = permutedims(tmp,[1,4,2,5,3,6])
            cores[d]    = reshape(tmp,sz[1]*sz[4],sz[2],sz[5],sz[3]*sz[6])
        end
        return TT(cores)
    end

    function *(ttm::TTm,a::Float64)
        # multiplication of TT with a scalar
        ttmcopy = copyTT(ttm)
        ttmcopy[1] = a*ttmcopy[1]

        return ttmcopy
    end

end

function outerproduct(tt1,tt2)
    D       = order(tt1)
    cores   = Vector{Array{Float64,4}}(undef,D)
    for d = 1:D
        sz1      = size(tt1[d])
        sz2      = size(tt2[d])
        tmp      = tt1[d][:]*tt2[d][:]'
        tmp      = reshape(tmp,Tuple(hcat(sz1...,sz2...)))
        tmp      = permutedims(tmp,[1,4,2,5,3,6])
        cores[d] = reshape(tmp,sz1[1]*sz2[1],sz1[2],sz2[2],sz1[3]*sz2[3])
    end

    return TT(cores)
end




#= multiplication of two TTm's, that returns a matrix, because only one core has a column index
    # assuming that sizes are [M1 M2 M3; 1 1 R], so final matrix is R x R
    function *(ttm1::TT,ttm2::TT)
        D       = order(ttm1);
        sz1     = size(ttm1[1]);
        sz2     = size(ttm2[1]);
        mat     = ttm1[1][1,:,1,:]'*ttm2[1][1,:,1,:]
        mat     = reshape(mat,1,sz1[4]*sz2[4])
        for d = 2:D-1
            sz1     = size(ttm1[d]);
            sz2     = size(ttm2[d]);

            tmp1    = reshape(permutedims(ttm1[d],[2,1,3,4]),(sz1[2],sz1[1]*sz1[4]))
            tmp2    = reshape(permutedims(ttm2[d],[2,1,3,4]),(sz2[2],sz2[1]*sz2[4]))

            tmp     = reshape(tmp1'*tmp2,sz1[1],sz1[4],sz2[1],sz2[4])

            tmp     = permutedims(tmp,[1,3,2,4])

            tmp     = reshape(tmp,sz1[1]*sz2[1],sz1[4]*sz2[4])
            mat     = mat*tmp
        end
        sz1     = size(ttm1[D]);
        sz2     = size(ttm2[D]);    
        tmp1    = reshape(mat,sz1[1],sz2[1])'*reshape(ttm1[D],sz1[1],sz1[2]*sz1[3])

        tmp1    = reshape(tmp1,sz2[1]*sz1[2],sz1[3])
        tmp2    = reshape(ttm2[D],sz2[1]*sz2[2],sz2[3])
        mat     = tmp1'*tmp2

        return mat
    end =#
    