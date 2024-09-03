using Pkg
Pkg.activate("onlineSKI")
using DelimitedFiles
using LinearAlgebra
using StatsBase
using Plots
using Random

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
includet("./functions/functionsTNSRKF_exactQR.jl")
using .functionsTNSRKF_exactQR

Random.seed!(1234)
dataX           = readdlm("./data/Volterra_inputsequence.csv",',');
datay           = readdlm("./data/Volterra_outputsequence.csv",',');
X               = dataX[1:1000];
y               = datay[1:1000];
Xtest           = dataX[1001:end];
ytest           = datay[1001:end];

# create synth data
N               = 1000;
Nstar           = 300;
mem             = 3;
D               = 7;

SNR             = 60;
norme           = norm(y)/10^(SNR/20);;
e               = randn(N,1);
e               = e/norm(e)*norme;
σ²              = var(e);
y               = y+e;

UUs             = Volterra_basisfuctions(vcat(X,Xtest),mem);
U               = UUs[1:1000,:];
Us              = UUs[1001:end,:];
Φ               = [U for _ in 1:D]; 
Φs              = [Us for _ in 1:D]; 
M               = Int.(mem*ones(D)) .+  1;

P0_             = ones(mem+1);
λtilde          = 0.2;
P0              = λtilde* [P0_ for _ in 1:D];

## Rw = 2

# Rw = 2, RL = 2, p = 4
mt,Lt,sv  = TNSRKF(N,y,Φ,[1,2,2,2,2,2,2,1],[1,2,2,2,2,2,2,1],M,2,P0,1,σ²,3);
rootmse_R2R2,nll_R2R2 = predictions(N,Nstar,Φs,mt,Lt,ytest);


# Rw = 2, RL = 4, p = 4
mt,Lt,sv  = TNSRKF(N,y,Φ,[1,2,2,2,2,2,2,1],[1,4,4,4,4,4,4,1],M,2,P0,1,σ²,3);
rootmse_R2R4,nll_R2R4 = predictions(N,Nstar,Φs,mt,Lt,ytest);

## Rw = 4
# Rw = 4, RL = 2, p = 4
mt_all,Lt_all  = TNSRKF(N,y,Φ,[1,4,4,4,4,4,4,1],[1,2,2,2,2,2,2,1],M,2,P0,2,σ²,3);
rootmse_R4R2,nll_R4R2 = predictions(N,Nstar,Φs,mt_all,Lt_all,ytest);

# Rw = 4, RL = 4, p = 4
mt_all,Lt_all  = TNSRKF(N,y,Φ,[1,4,4,4,4,4,4,1],[1,4,4,4,4,4,4,1],M,2,P0,2,σ²,3);
rootmse_R4R4,nll_R4R4 = predictions(N,Nstar,Φs,mt_all,Lt_all,ytest);


##
pgfplotsx()
using LaTeXStrings

plt1 = plot(rootmse_R2R2,linewidth=2,xlabel=L"$t$",ylabel="RMSE",label=L"$R_\mathbf{w} = 2, R_\mathbf{L} = 2$",legend=:topright,ytickfontsize=20,xtickfontsize=20,xguidefontsize=20,yguidefontsize=20);
plot!(rootmse_R4R2 ,linewidth=2,xlabel=L"$t$",label=L"$R_\mathbf{w} = 4, R_\mathbf{L} = 2$");
plot!(rootmse_R2R4 ,linewidth=2,xlabel=L"$t$",label=L"$R_\mathbf{w} = 2, R_\mathbf{L} = 4$");
plot!(rootmse_R4R4 ,linewidth=2,xlabel=L"$t$",label=L"$R_\mathbf{w} = 4, R_\mathbf{L} = 4$");
plot!(legendfontsize=14);
ylims!(0, 3);
plt2 = plot(nll_R2R2     ,linewidth=2,xlabel=L"$t$",ylabel="NLL",legend=false,ytickfontsize=20,xtickfontsize=20,xguidefontsize=20,yguidefontsize=20);
plot!(nll_R2R4     ,linewidth=2);
plot!(nll_R4R2     ,linewidth=2);
plot!(nll_R4R4     ,linewidth=2);
plot!(legendfontsize=20);
ylims!(-300, 2000);
plt = plot(plt1,plt2,layout=(1,2))
savefig(plt,"./figures/synth_ranks.png")


rootmse_R2R1[end]
rootmse_R2R2[end]
rootmse_R2R3[end]
rootmse_R2R4[end]

rootmse_R4R1[end]
rootmse_R4R2[end]
rootmse_R4R3[end]
rootmse_R4R4[end]

nll_R2R1[end]
nll_R2R2[end]
nll_R2R3[end]
nll_R2R4[end]

nll_R4R1[end]
nll_R4R2[end]
nll_R4R3[end]
nll_R4R4[end]
