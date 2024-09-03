module functionsBasic

using LinearAlgebra, SparseArrays

export gensynthdata, covSE, synthdata_Volterra, Volterra_basisfuctions, gensynthdata_nonlinfun

    function gensynthdata(N::Int64,D::Int64,hyp::Vector)
        σ_n   = sqrt(hyp[3]);
        X     = zeros(N,D);
        jitter = sqrt(eps(1.))
        for d = 1:D
            for n = 1:N
                X[n,d] = rand(1)[1].* 2 .-1
            end
        end
        K      = covSE(X,X,hyp);
        f      = Matrix(cholesky(K+jitter*Matrix(I,size(K))))*randn(N);
        y      = f + σ_n*randn(size(f,1));
        return X, y, f, K 
    end

    function covSE(Xp::Matrix{Float64},Xq::Matrix{Float64},hyp::Vector{Float64})
        ℓ     = hyp[1];
        σ_f   = hyp[2];
        D     = size(Xp,2)

        K = zeros(size(Xp,1),size(Xq,1))
        for i = 1:size(Xp,1)
            for j = 1:size(Xq,1)
                exparg = norm(Xp[i,:]-Xq[j,:])^2/2ℓ
                K[i,j] = σ_f * exp(-exparg)
            end
        end
        return K
    end

    function synthdata_Volterra(mem,N,Nstar)
        H       = zeros(mem+1,100);
        for i=1:100
            H[:,i]=abs.(1e0*randn(1)).*exp.(-collect((rand(1:10).*[1:mem+1])[1]));
        end

        u       = randn(N+Nstar);
        U       = Volterra_basisfuctions(u,mem);
        y       = sum((U*H).^D,dims=2);

        return u,y
    end

    function Volterra_basisfuctions(u,mem)
    # u     input sequence
    # mem     Volterra memory
    # case for d=1 and p=1

        N       = size(u,1);
        U       = zeros(N,mem+1);
        u       = [zeros(mem-1); u];

        for i = mem:N+mem-1            
            temp = ones(1,mem+1);
            for j=1:mem
                temp[j+1:2+j-1] = u[i-j+1,:];
            end    
            U[i-mem+1,:] = temp; # if d>1 mkron needed
        end

        return U
    end

    function gensynthdata_nonlinfun(D, N, σ_y²)
        # D: input dimension
        # N: number of data points
        # σ_y²: noise variance for the observations
        
        # Generate input data
        X = rand(N, D) .- 0.5
        
        # Define a nonlinear function
        function nonlinear_function(X)
            return sin.(2π * sum(X, dims=2)) .+ sum(X .^ 2, dims=2)
        end
        
        # Generate the synthetic output data
        y_clean = nonlinear_function(X)
        
        # Add Gaussian noise to the output data
        y = y_clean .+ σ_y² * randn(N)
        
        return X, y
    end

end