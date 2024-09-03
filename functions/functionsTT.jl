module functionsTT

using LinearAlgebra
import Base: transpose 

export TT, TTv,TTm, size, norm, order, rank, 
       shiftTTnorm, copyTT, overwriteTT!,
       ttv2vec, ttm2mat, ttm2ttv, ttv2ttm, 
       I_TTm,initTT,
       TT_SVD,TTv_SVD, TTm_SVD, TT_ALS, roundTT,
       tmult

mutable struct TT{N}
    cores::Vector{Array{Float64,N}}
    normcore::Int64
    function TT(cores::Vector{Array{Float64,N}},normcore::Int64) where N
        new{ndims(cores[1])}(cores,normcore) 
    end
end
TT(cores) = TT(cores,0);

# aliases for MPS and MPO
const TTv = TT{3};
const TTm = TT{4};

# indexing to get and set a core in an TT
Base.IndexStyle(::Type{<:TT}) = IndexLinear() 
Base.getindex(tt::TT, i::Int) = tt.cores[i] # tt[1] gives the first core
Base.getindex(tt::TT, range::UnitRange{Int64}) = [tt.cores[i] for i in range]
Base.setindex!(tt::TT,v,i::Int) = setindex!(tt.cores, v, i) # tt[1] = rand(1,5,3) sets the first core

function Base.size(tt::TTv)
    [size(core)[2] for core in tt.cores];
end

function Base.size(ttm::TTm)
    sizes = Int.(zeros(2,order(ttm)))
    for d = 1:order(ttm)
        sizes[1,d] = size(ttm.cores[d],2)
        sizes[2,d] = size(ttm.cores[d],3)
    end
    return sizes
end

function order(tt::TT)
    collect(size(tt.cores))[1]
end

function LinearAlgebra.rank(tt::TT)
    sizes = [size(core) for core in tt.cores];
    [[sizes[i][1] for i in 1:length(sizes)]..., 1]    
end

function LinearAlgebra.norm(tt::TT{3})
    D       = order(tt);
    tmp     = reshape(tt[1],size(tt[1],2),size(tt[1],3))
    ttnorm  = reshape(tmp'*tmp,1,size(tt[1],3)^2)
    for d = 2:D
        tmp     = reshape(permutedims(tt[d],[2,1,3]),size(tt[d],2),size(tt[d],1)*size(tt[d],3))
        tmp     = reshape(tmp'*tmp,size(tt[d],1),size(tt[d],3),size(tt[d],1),size(tt[d],3))
        tmp     = permutedims(tmp,[1,3,2,4])
        ttnorm  = ttnorm*reshape(tmp,size(tt[d],1)^2,size(tt[d],3)^2)
    end
    return sqrt(ttnorm[1])
end

function LinearAlgebra.norm(tt::TT{4})
    ttv = ttm2ttv(tt)
    return norm(ttv)
end

function shiftTTnorm(tt::TTv,d::Int64,dir::Int64)

    if dir == 1
        sztt    = size(tt[d])
        Gl      = reshape(tt[d],sztt[1]*sztt[2],sztt[3])
        F       = qr(Gl);
        R       = Matrix(F.R); Q = Matrix(F.Q);
        sztt    = size(tt[d+1])
        tt[d+1] = reshape(R*reshape(tt[d+1],sztt[1],sztt[2]*sztt[3]),sztt)
    elseif dir == -1
        sztt    = size(tt[d])
        Gr      = reshape(tt[d],sztt[1],sztt[2]*sztt[3])
        F       = qr(Gr');
        R       = Matrix(F.R)'; Qt = Matrix(F.Q);
        Q       = Qt';
        sztt    = size(tt[d-1])
        tt[d-1] = reshape(reshape(tt[d-1],sztt[1]*sztt[2],sztt[3])*R,sztt)
    end

    tt[d]       = reshape(Q, size(tt[d]));
    tt.normcore = tt.normcore + dir; 
    return tt
end

function overwriteTT!(oldtt,newtt)
    D = order(oldtt)
    for d = 1:D
        oldtt[d] = newtt[d]
    end
end

function copyTT(tt)
    D = order(tt)
    cores = Vector{Array{Float64,ndims(tt[1])}}(undef,D)
    for d = 1:D
        cores[d] = tt[d]
    end
    return TT(cores,tt.normcore)
end

function transpose(ttm::TTm)
    D     = order(ttm);
    cores = Vector{Array{Float64,4}}(undef,D);
    for d = 1: order(ttm)
        cores[d] = permutedims(ttm[d],[1,3,2,4]);
    end
    return TT(cores,ttm.normcore)
end

function transpose(tt::TTv)
    D     = order(tt);
    cores = Vector{Array{Float64,4}}(undef,D);
    for d = 1: D
        sz       = size(tt[d])
        cores[d] = reshape(tt[d],sz[1],1,sz[2],sz[3]);
    end
    return TT(cores,tt.normcore)
end

# reconstruction of vector represented by an TTv
function ttv2vec(tt::TTv)
    tensor = reshape(tt[1],size(tt[1],1)*size(tt[1],2),size(tt[1],3))
    for i = 2:order(tt)
        tensor = tensor*reshape(tt[i],size(tt[i],1),size(tt[i],2)*size(tt[i],3))
        tensor = reshape(tensor, (Int(length(tensor)/size(tt[i],3)), size(tt[i],3)));
    end
    vector = tensor
    return vector
end

# reconstruction of matrix represented by an TTm
function ttm2mat(ttm::TTm)
    tensor = reshape(ttm[1],size(ttm[1],1)*size(ttm[1],2)*size(ttm[1],3),size(ttm[1],4))
    sizes  = size(ttm);
    D      = order(ttm)

    for i = 2:D
        tensor = tensor*reshape(ttm[i],size(ttm[i],1),size(ttm[i],2)*size(ttm[i],3)*size(ttm[i],4))
        tensor = reshape(tensor, (Int(length(tensor)/size(ttm[i],4)), size(ttm[i],4)));
    end
    tensor = reshape(tensor,(sizes[:]...));
    tensor = permutedims(tensor,[collect(1:2:2D-1)..., collect(2:2:2D)...]);
    matrix = reshape(tensor,(prod(sizes[1,:]),prod(sizes[2,:])));
    return matrix
end

function ttm2ttv(ttm)
    D       = order(ttm)
    cores   = Vector{Array{Float64,3}}(undef,D)
    for d = 1:D
        s        = size(ttm[d])
        cores[d] = reshape(ttm[d],s[1],s[2]*s[3],s[4])
    end
    return TT(cores,ttm.normcore)
end

function ttv2ttm(ttv,sizes)
    D       = order(ttv)
    cores   = Vector{Array{Float64,4}}(undef,D)
    for d = 1:D
        cores[d] = reshape(ttv[d],size(ttv[d],1),sizes[1,d],sizes[2,d],size(ttv[d],3))
    end
    return TT(cores,ttv.normcore)
end

function I_TTm(middlesizes::Matrix{Int64},factor)

    cores = Vector{Array{Float64,4}}(undef,size(middlesizes,2));
    for i = 1:size(middlesizes,2)
        core = Matrix(I,(middlesizes[1,i],middlesizes[2,i]));
        core = reshape(core, (1, middlesizes[1,i],middlesizes[2,i], 1) );
        cores[i] = convert(Array{Float64}, core);
    end
    cores[1] = cores[1]*factor

    TT(cores)
end

function TTm_SVD(mat::Matrix,middlesizes::Matrix,acc::Float64)
    sizes   = Tuple(reshape(middlesizes',(length(middlesizes),1)));
    tensor  = reshape(mat,sizes);
    permind = [ (i-1)*size(middlesizes,2)+j for j in 1:size(middlesizes,2) for i in 1:2 ];
    tensor  = permutedims(tensor,permind);
    resind  = Tuple([prod(col) for col in eachcol(middlesizes)]);
    tensor  = reshape(tensor,resind)
    tt,err  = TT_SVD(tensor,acc);
    rnks    = rank(tt);
    return TT( [reshape(tt[i],(rnks[i], middlesizes[:,i]..., rnks[i+1])) for i = 1:order(tt)] ),err
end

function TTv_SVD(vec::Vector,middlesizes::Vector,acc::Float64)
    tensor  = reshape(vec,Tuple(middlesizes));
    return TT_SVD(tensor,acc);
end

function TT_SVD(tensor::Array{Float64},ϵ::Float64)
    ########################################################################    
    #   Computes the cores of a TT for the given tensor and accuracy (acc)     
    #   Resources:
    #   V. Oseledets: Tensor-Train Decomposition, 2011, p.2301: Algorithm 1
    #   April 2021, Clara Menzen
    ########################################################################
        D           = ndims(tensor);
        cores       = Vector{Array{Float64,3}}(undef,D);
        frobnorm    = norm(tensor); 
    
        δ           = ϵ / sqrt(D-1) * frobnorm;
        err2        = 0;
        rprev       = 1;
        sizes       = size(tensor);
        C           = reshape( tensor, (sizes[1], Int(length(tensor) / sizes[1]) ));
        for k = 1 : D-1
            # truncated svd 
            F   = svd!(C); 
            rcurr = length(F.S);
    
            sv2 = cumsum(reverse(F.S).^2);
            tr  = Int(findfirst(sv2 .> δ^2))-1;
            if tr > 0
                rcurr = length(F.S) - tr;
                err2 += sv2[tr];
            end
            
            # new core
            cores[k] = reshape(F.U[:,1:rcurr],(rprev,sizes[k],rcurr));
            rprev    = rcurr;
            C        = Diagonal(F.S[1:rcurr])*F.Vt[1:rcurr,:];
            C        = reshape(C,(rcurr*sizes[k+1], Int(length(C) / (rcurr*sizes[k+1])) ) );
        end
        cores[D] = reshape(C,(rprev,sizes[D],1));
        return TT(cores,D), sqrt(err2)/frobnorm
    end

    function TT_SVD(tensor::Array{Float64},ϵ::Float64,dd)
            # computes low rank Cholesky factor from vectorized matrix
            D           = ndims(tensor);
            cores       = Vector{Array{Float64,4}}(undef,dd);
            frobnorm    = norm(tensor); 
        
            δ           = ϵ / sqrt(D-1) * frobnorm;
            err2        = 0;
            rprev       = 1;
            sqrtS       = zeros(2,2);
            sizes       = size(tensor);
            C           = reshape( tensor, (sizes[1], Int(length(tensor) / sizes[1]) ));
            for k = 1 : dd
                # truncated svd 
                F   = svd!(C); 
                rcurr = length(F.S);
        
                sv2 = cumsum(reverse(F.S).^2);
                tr  = Int(findfirst(sv2 .> δ^2))-1;
                if tr > 0
                    rcurr = length(F.S) - tr;
                    err2 += sv2[tr];
                end
                
                # new core
                cores[k] = reshape(F.U[:,1:rcurr],(rprev,sizes[k],1,rcurr));
                rprev    = rcurr;
                C        = Diagonal(F.S[1:rcurr])*F.Vt[1:rcurr,:];
                sqrtS    = diagm(sqrt.(F.S[1:rcurr]))
                C        = reshape(C,(rcurr*sizes[k+1], Int(length(C) / (rcurr*sizes[k+1])) ) );
            end
            sz        = size(cores[dd])
            tmp       = reshape(cores[dd],sz[1]*sz[2]*sz[3],sz[4])*sqrtS;
            cores[dd] = permutedims(reshape(tmp,sz[1],sz[2],sz[3],sz[4]),[1,2,4,3])
            return TT(cores,dd), sqrt(err2)/frobnorm
        end

# TT-ALS ###############################################################################
# Comments:
# ALS without orthog is really slow (probably getUTU needs to be optimized) 
# and almost never used unless an initial tt is inputted which is not site-k

# no initial tt, automatically with orthogonalization
function TT_ALS(tensor::Array{Float64},rnks::Vector{Int64})
    D     = ndims(tensor);
    sizes = size(tensor);
    cores = Vector{Array{Float64,3}}(undef,D);
    for i = 1:D-1 # creating site-D canonical initial tensor train
        tmp = qr(rand(rnks[i]*sizes[i], rnks[i+1]));
        cores[i] = reshape(Matrix(tmp.Q),(rnks[i], sizes[i], rnks[i+1]));
    end
    cores[D] = reshape(rand(rnks[D]*sizes[D]),(rnks[D], sizes[D], 1));
    tt0 = TT(cores,D);
    return TT_ALS(tensor,tt0)
end


# with / without orthogonalization
function TT_ALS(tensor::Array{Float64},tt::TTv)
    maxiter = 200;
    N       = order(tt);
    rnks    = rank(tt);
    sizes   = size(tt);

    for i = 1:maxiter
        for k = 1:2N-2
            if tt.normcore == 0
                swipe = [collect(1:N)..., collect(N-1:-1:2)...];
                n     = swipe[k];
                UTU   = getUTU(tt,n);
                UTy   = getUTy(tt,tensor,n);
                tt[n] = reshape(inv(UTU)*UTy,(rnks[n],sizes[n],rnks[n+1]));
            else
                swipe = [collect(N:-1:2)..., collect(1:N-1)...];
                Dir   = Int.([-ones(1,N-1)...,ones(1,N-1)...]);
                n     = swipe[k];
                UTy   = getUTy(tt,tensor,n);
                tt[n] = reshape(UTy,(rnks[n],sizes[n],rnks[n+1]));
                shiftTTnorm(tt,n,Dir[k])
            end
        end
    end
    return tt
end


# TT-ALS for a vector without initial tt
function TT_ALS(vector::Vector{Float64},middlesizes::Matrix{Int64},rnks::Vector{Int64})
    tensor = reshape(vector,Tuple(middlesizes));
    return TT_ALS(tensor,rnks);
end


# TT-ALS for vector with initial TT
function TT_ALS(vector::Vector{Float64},tt0::TTv)
    tensor = reshape(vector,Tuple([size(tt0)[i][1] for i = 1:order(tt0)]));
    return TT_ALS(tensor,tt0);  
end


##########################################
# functions for ALS with orthogonalization
function getUTy(tt::TTv,tensor,n::Int64)
    N     = order(tt);
    sizes = size(tensor);
    rnks  = rank(tt);
    if n == N 
        Gleft    = supercores(tt,N);
        newsizes = (prod(sizes[1:N-1]), sizes[N]);
        UTy      = Gleft*reshape(tensor,Tuple(newsizes));
    elseif n == 1
        Gright   = supercores(tt,1);
        newsizes = (sizes[1], prod(sizes[2:N]));
        UTy      = reshape(tensor,Tuple(newsizes))*Gright;
    else
        Gleft, Gright = supercores(tt,n);
        newsizes1     = (prod(sizes[1:n-1]), prod(sizes[n:N]));
        tmp           = Gleft*reshape(tensor,newsizes1);
        newsizes2     = (rnks[n][1]*sizes[n], prod(sizes[n+1:N]));
        UTy           = reshape(tmp,newsizes2)*Gright;
    end
    return UTy[:]
end

function supercores(tt::TTv,n::Int64)
    D     = order(tt);
    sizes = size(tt);
    rnks  = rank(tt);
    if  n == 1
        Gright = reshape(tt[2],rnks[2]*sizes[2],rnks[3])
        for i = 3:D
            Gright = Gright*reshape(tt[i],rnks[i],sizes[i]*rnks[i+1]);
            Gright = reshape(Gright,rnks[2]*prod(sizes[2:i]),rnks[i+1])
        end
        return reshape(Gright,rnks[2],prod(sizes[2:D]))'
    elseif n == D
        Gleft = tt[1][1,:,:];
        for i = 2:D-1
            Gleft = Gleft*reshape(tt[i],rnks[i],sizes[i]*rnks[i+1]);
            Gleft = reshape(Gleft,prod(sizes[1:i]),rnks[i+1]);
        end
        return Gleft'
    else
        Gleft = tt[1][1,:,:];
        for i = 2:n-1
            Gleft = Gleft*reshape(tt[i],rnks[i],sizes[i]*rnks[i+1]);
            Gleft = reshape(Gleft,prod(sizes[1:i]),rnks[i+1]);
        end

        Gright = reshape(tt[n+1],rnks[n+1]*sizes[n+1],rnks[n+2])
        for i = n+2:D
            Gright = Gright*reshape(tt[i],rnks[i],sizes[i]*rnks[i+1]);
            Gright = reshape(Gright,rnks[2]*prod(sizes[2:i]),rnks[i+1])
        end
        Gright = reshape(Gright,rnks[2],prod(sizes[n+1:D]))

        return Gleft',Gright'
    end
end

# function for ALS without orthogonalization
function getUTU(tt::TTv,n::Int64)
    N     = order(tt);
    sizes = size(tt);
    rnks  = rank(tt);

    Gleft = [1];
    for i = 1:n-1
        Gleft = Gleft * contractcores(tt[i],tt[i]);
    end
    Gleft = reshape(Gleft,(rnks[n][1],rnks[n][1]));

    Gright = [1];
    for i = N:-1:n+1
        Gright = contractcores(tt[i],tt[i]) * Gright;
    end
    Gright = reshape(Gright,(rnks[n][2],rnks[n][2]));

    return kron(kron(Gright, 1.0*Matrix(I,sizes[n][1],sizes[n][1])), Gleft)
end

function roundTT(tt::TT{3}, ϵ::Float64)

    D           = order(tt);
    cores       = Vector{Array{Float64,3}}(undef,D);
    middlesizes = size(tt);
    frobnorm    = norm(tt); 
    err2        = 0;
    
    # Right-to-left orthogonalization 
        G       = tt[D];
        for d = D:-1:2
            szd      = size(tt[d])
            G        = reshape(G,szd[1],szd[2]*size(G,3))
            F        = lq(G);
            Q        = Matrix(F.Q);
            L        = Matrix(F.L);
            G        = reshape(Q, ( size(Q,1), middlesizes[d], Int(length(Q)/(size(Q,1)*middlesizes[d])) ) );
            cores[d] = G;
            szdd     = size(tt[d-1])
            G        = reshape(reshape(tt[d-1],szdd[1]*szdd[2],szdd[3])*L,szdd[1],szdd[2],size(L,2))
        end
        cores[1] = G;
        # Compression of the orthogonalized representation
        δ = ϵ / sqrt(D-1) * frobnorm;
        for d = 1:D-1
            szd   = size(cores[d])
            F     = svd(reshape(cores[d],szd[1]*szd[2],szd[3]));
            rcurr = length(F.S);
            sv2   = cumsum(reverse(F.S).^2);
            tr    = Int(findfirst(sv2 .> δ^2))-1;
            if tr > 0
                rcurr = length(F.S) - tr;
                err2 += sv2[tr];
            end
            Utr = F.U[:,1:rcurr];
            Str = F.S[1:rcurr];
            Vtr = F.Vt[1:rcurr,:];

            cores[d]   = reshape( Utr,( Int(length(Utr)/(size(cores[d],2)*rcurr)), size(cores[d],2), rcurr ) );
            
            szdd       = size(cores[d+1])
            cores[d+1] = reshape(Matrix((Diagonal(Str)*Vtr)) * reshape(cores[d+1],szdd[1],szdd[2]*szdd[3]),rcurr,szdd[2:end]...)

        end
        return TT(cores,D);
    end

    function roundTT(ttm::TT{4}, ϵ::Float64)
        tt = ttm2ttv(ttm);
        tt = roundTT(tt, ϵ)
        return ttv2ttm(tt,size(ttm))
    end

    function roundTT(ttm::TT{4}, ϵ::Float64,dd)
        tt = ttm2ttv(ttm);
        tt = roundTT(tt, ϵ)
        for d = order(tt):-1:dd+1
            tt = shiftTTnorm(tt,d,-1)
        end
        return ttv2ttm(tt,size(ttm))
    end

    function roundTT(tt::TT{3},rnks)
        D           = order(tt);
        cores       = Vector{Array{Float64,3}}(undef,D);
        middlesizes = size(tt);
    
    # Right-to-left orthogonalization 
        G       = tt[D];
        for d = D:-1:2
            szd      = size(tt[d])
            G        = reshape(G,szd[1],szd[2]*size(G,3))
            F        = lq(G);
            Q        = Matrix(F.Q);
            L        = Matrix(F.L);
            G        = reshape(Q, ( size(Q,1), middlesizes[d], Int(length(Q)/(size(Q,1)*middlesizes[d])) ) );
            cores[d] = G;
            szdd     = size(tt[d-1])
            G        = reshape(reshape(tt[d-1],szdd[1]*szdd[2],szdd[3])*L,szdd[1],szdd[2],size(L,2))
        end
        cores[1] = G;
        # Compression of the orthogonalized representation
        for d = 1:D-1
            szd   = size(cores[d])
            F     = svd(reshape(cores[d],szd[1]*szd[2],szd[3]));

            if size(F.U,2) < rnks[d+1]
                ranktr = size(F.U,2)
            else
                ranktr = rnks[d+1]
            end
            Utr = F.U[:,1:ranktr];
            Str = F.S[1:ranktr];
            Vtr = F.Vt[1:ranktr,:];

            cores[d]   = reshape( Utr,( Int(length(Utr)/(size(cores[d],2)*ranktr)), size(cores[d],2), ranktr ) );
            
            szdd       = size(cores[d+1])
            cores[d+1] = reshape(Matrix((Diagonal(Str)*Vtr)) * reshape(cores[d+1],szdd[1],szdd[2]*szdd[3]),ranktr,szdd[2:end]...)

        end
        return TT(cores,D);
    end

    function roundTT(ttm::TT{4}, rnks::Vector)
        tt = ttm2ttv(ttm);
        tt = roundTT(tt, rnks)
        return ttv2ttm(tt,size(ttm))
    end

    function roundTT(ttm::TT{4}, rnks::Vector,dd)
        tt = ttm2ttv(ttm);
        tt = roundTT(tt, rnks)
        for d = order(tt):-1:dd+1
            tt = shiftTTnorm(tt,d,-1)
        end
        return ttv2ttm(tt,size(ttm))
    end

    function tmult(ten1,ten2,modes1,modes2)
        sz1  = size(ten1)
        sz2  = size(ten2)
        sz11 = sz1[modes1[1]]
        sz12 = sz1[modes1[2]]
        sz21 = sz2[modes2[1]]
        sz22 = sz2[modes2[2]]
    
        perm1 = vcat(modes1[1],modes1[2]...)
        perm2 = vcat(modes2[1],modes2[2]...)
        mat1  = reshape(permutedims(ten1,perm1),(prod(sz11),prod(sz12)))
        mat2  = reshape(permutedims(ten2,perm2),(prod(sz21),prod(sz22)))
    
        return reshape(mat1*mat2,Tuple(vcat(sz11...,sz22...)))
    end

    function initTT(rnks,M,dd,D)
        # create site-d canonical initial tensor train    
        cores = Vector{Array{Float64,3}}(undef,D);
        for d = 1:dd-1 
            tmp         = qr(rand(rnks[d]*M[d], rnks[d+1]));
            cores[d]    = reshape(Matrix(tmp.Q),(rnks[d], M[d], rnks[d+1]));
        end
        cores[dd]       = reshape(randn(rnks[dd]*M[dd]*rnks[dd+1]),(rnks[dd], M[dd], rnks[dd+1]))
        for d = dd+1:D
            tmp         = qr(rand(M[d]*rnks[d+1],rnks[d]));
            cores[d]    = reshape(Matrix(tmp.Q)',(rnks[d], M[d], rnks[d+1]));
        end
        return TT(cores,dd);
    end
    
    function initTT(rnks,M,D)  
        cores = Vector{Array{Float64,3}}(undef,D);
        for d = 1:D 
            cores[d]    = randn(rnks[d], M[d], rnks[d+1])
        end
    
        return TT(cores);
    end

end

