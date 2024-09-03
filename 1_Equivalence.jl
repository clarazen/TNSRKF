using Pkg
Pkg.activate("onlineSKI")
using LinearAlgebra
using Plots
using Random
using Distributions

includet("./functions/functionsBasic.jl")
using .functionsBasic
includet("./functions/functionsTT.jl")
using .functionsTT
includet("./functions/functions_KhatriRao_Kronecker.jl")
using .functions_KhatriRao_Kronecker
includet("./functions/functionsBasisFunctions.jl")
using .functionsBasisfunctions
includet("./functions/functionsTTmatmul.jl")
using .functionsTTmatmul
includet("./functions/functionsbasicoperations.jl")
using .functionsbasicoperations
includet("./functions/functionsTNKF.jl")
using .functionsTNKF
includet("./functions/functionsTNSRKF_exactQR.jl")
using .functionsTNSRKF_exactQR

#includet("./functions/functionsTNSRKF.jl")
#using functionsTNSRKF

# generate syntehtic data
Random.seed!(1234)
N               = 100;
Nstar           = 100;
D               = 3;
hyp             = [0.1, 1, 0.01];

data     = zeros(N+Nstar,D);
for d = 1:D
    for n = 1:N+Nstar
        data[n,d] = rand(1)[1].* 2 .-1
    end
end
X               = data[1:N,:];
Xtest           = data[N+1:end,:];

M               = [4,4,4];
# generate basis functions
boundsMin   = minimum(X,dims=1);
boundsMax   = maximum(X,dims=1);
L           = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*hyp[1] ;
# SE
Φ           = Vector{Matrix{Float64}}(undef,D);
Λ           = Vector{Vector}(undef,D);
basisfunctionsSE!(M,X,hyp[1],hyp[2],L,Φ,Λ);
Φs          = Vector{Matrix{Float64}}(undef,D);
Λs          = Vector{Vector}(undef,D);
basisfunctionsSE!(M,Xtest,hyp[1],hyp[2],L,Φs,Λs);

mvn = MvNormal(zeros(64), ttm2mat(factors2ttm(diagm.(Λ))));
ytot   = vcat(ttm2mat(khr2ttm(Φ))',ttm2mat(khr2ttm(Φs))')*rand(mvn,1);

y               = ytot[1:N];
ytest           = ytot[N+1:end];


## Kalman filter reference
mstar_kf,σstar_kf,mt_kf,Pt_kf  = loopKF(M,N,Nstar,Φ,Φs,y,hyp[3],Λ);
rootmse_kf          = [sqrt(sum((mstar_kf[:,t] - ytest).^2/Nstar)) for t=1:N]
nll_kf              = [sum(.5*(log.(2π*σstar_kf[:,t].^2) + (ytest - mstar_kf[:,t]).^2 ./ σstar_kf[:,t].^2)) for t=1:N]


## TN algorithms
maxiter         = 1;
dd              = 2;

# Rw = 4, RL = 16, p = 1
p = 1;
mt,Lt,sv = TNSRKF(N,y,Φ,[1,4,4,1],[1,16,16,1],M,2,Λ,1,hyp[3],p);
rootmse_p1,nll_p1,mstar_p1,σstar_p1= predictions(N,Nstar,Φs,mt,Lt,ytest);

# Rw = 4, RL = 16, p = 2
p = 2;
mt,Lt,sv = TNSRKF(N,y,Φ,[1,4,4,1],[1,16,16,1],M,2,Λ,1,hyp[3],p);
rootmse_p2,nll_p2,mstar_p2,σstar_p2 = predictions(N,Nstar,Φs,mt,Lt,ytest);

# Rw = 4, RL = 16, p = 4
p = 4;
mt,Lt,sv = TNSRKF(N,y,Φ,[1,4,4,1],[1,16,16,1],M,2,Λ,1,hyp[3],p);
rootmse_p4,nll_p4,mstar_p3,σstar_p3 = predictions(N,Nstar,Φs,mt,Lt,ytest);

# Rw = 4, RL = 16, p not bounded
p = 100;
mt,Lt,sv = TNSRKF(N,y,Φ,[1,4,4,1],[1,16,16,1],M,2,Λ,1,hyp[3],p);
rootmse_p8,nll_p8,mstar_p4,σstar_p4 = predictions(N,Nstar,Φs,mt,Lt,ytest);



rootmse_kf[end]
nll_kf[end]

rootmse_p1[end]
nll_p1[end]
rootmse_p2[end]
nll_p2[end]
rootmse_p4[end]
nll_p4[end]
rootmse_p8[end]
nll_p8[end]
