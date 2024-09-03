using Pkg
Pkg.activate("onlineSKI")
using LinearAlgebra
using Plots
using DelimitedFiles
using StatsBase

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

X       = readdlm("./data/X_cascaded.csv",',');
y       = readdlm("./data/y_cascaded.csv",',');
Xtest   = readdlm("./data/Xtest_cascaded.csv",',');
ytest   = readdlm("./data/ytest_cascaded.csv",',');

meany           = mean(y);
y               = y .- meany;
ytest           = ytest .- meany;

function scale_to_min_minus_one_max_one(matrix)
    col_min = minimum(matrix, dims=1) # Find minimum value in each column
    col_max = maximum(matrix, dims=1) # Find maximum value in each column
    range = col_max .- col_min # Calculate the range of each column

    scaled_matrix = (matrix .- col_min) ./ range # Scale each element to [0, 1]
    scaled_matrix = 2 * scaled_matrix .- 1 # Map values from [0, 1] to [-1, 1]

    return scaled_matrix
end

X               = scale_to_min_minus_one_max_one(X);
Xtest           = scale_to_min_minus_one_max_one(Xtest);
Nstar           = size(Xtest,1);

hyp             = [0.2305,0.0168,5.5115e-06];

N,D             = size(X);
M               = [4,4,4,4,4,4,4,4,4,4,4,4,4,4];

boundsMin       = minimum(X,dims=1);
boundsMax       = maximum(X,dims=1);
L               = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*hyp[1] ;
# SE
Φ           = Vector{Matrix{Float64}}(undef,D);
Λ           = Vector{Vector}(undef,D);
basisfunctionsSE!(M,X,hyp[1],hyp[2],L,Φ,Λ);
Φs          = Vector{Matrix{Float64}}(undef,D);
Λs          = Vector{Vector}(undef,D);
basisfunctionsSE!(M,Xtest,hyp[1],hyp[2],L,Φs,Λs);

factors = sqrt.(diagm.(Λ));

rnks_mean = [1,4,10,10,10,10,10,10,10,10,10,10,10,4,1];

## RL = 1
rnks_cova_1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
# TNKF RP = 1
mt,Pt = loopTNKF(y,Φ,N,M,D,Λ,rnks_mean,rnks_cova_1,10,hyp[3]);
rootmse_tnkf_1,nll_tnkf_1 = predictions_TNKF(N,Nstar,Φs,mt,Pt,ytest[33:end],Xtest);
# TNSRKF PL = 1
mt,Lt,sv =  TNSRKF(N,y,Φ,rnks_mean,rnks_cova_1,M,10,Λ,1,hyp[3],4);
rootmse_tnsrkf_1,nll_tnsrkf_1,mstar_1,σstar_1 = predictions(N,Nstar,Φs,mt,Lt,ytest[33:end]);

## RL = 2
rnks_cova_2 = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,1];
# TNKF RP = 4
mt,Pt = loopTNKF(y,Φ,N,M,D,Λ,rnks_mean,rnks_cova_2.^2,10,hyp[3]);
rootmse_tnkf_2,nll_tnkf_2 = predictions_TNKF(N,Nstar,Φs,mt,Pt,ytest[33:end],Xtest);
# TNSRKF RL = 2
mt,Lt,sv =  TNSRKF(N,y,Φ,rnks_mean,rnks_cova_2,M,10,Λ,1,hyp[3],4);
rootmse_tnsrkf_2,nll_tnsrkf_2,mstar_2,σstar_2 = predictions(N,Nstar,Φs,mt,Lt,ytest[33:end]);

## RL = 3
rnks_cova_3 = [1,3,3,3,3,3,3,3,3,3,3,3,3,3,1];
# TNKF RP = 9
mt,Pt = loopTNKF(y,Φ,N,M,D,Λ,rnks_mean,rnks_cova_3.^2,10,hyp[3]);
rootmse_tnkf_3,nll_tnkf_3 = predictions_TNKF(N,Nstar,Φs,mt,Pt,ytest[33:end],Xtest);
# TNSRKF RL = 3
@run mt,Lt,sv =  TNSRKF(N,y,Φ,rnks_mean,rnks_cova_3,M,10,Λ,1,hyp[3],4);
rootmse_tnsrkf_3,nll_tnsrkf_3,mstar_3,σstar_3 = predictions(N,Nstar,Φs,mt,Lt,ytest[33:end]);


## RL = 4
rnks_cova_4 = [1,4,4,4,4,4,4,4,4,4,4,4,4,4,1];
# TNKF RP = 16
mt,Pt = loopTNKF(y,Φ,N,M,D,Λ,rnks_mean,rnks_cova_4.^2,10,hyp[3]);
rootmse_tnkf_4,nll_tnkf_4 = predictions_TNKF(N,Nstar,Φs,mt,Pt,ytest[33:end],Xtest);
# TNSRKF RL = 4
mt,Lt,sv =  TNSRKF(N,y,Φ,rnks_mean,rnks_cova_4,M,10,Λ,1,hyp[3],4);
rootmse_tnsrkf_4,nll_tnsrkf_4,mstar_4,σstar_4 = predictions(N,Nstar,Φs,mt,Lt,ytest[33:end]);


## Plots
pgfplotsx()
using LaTeXStrings

plt1 = plot(rootmse_tnsrkf_1,label=L"\mathrm{TNSRKF},\;\;R_\mathbf{L}=1 ")
plot!(rootmse_tnkf_1,label=L"\mathrm{TNKF},\;\;R_\mathbf{L}=1 ")
#plot!(rootmse_tnsrkf_2,label=L"\mathrm{TNSRKF},\;\;R_\mathbf{L}=2 ")
#plot!(rootmse_tnkf_2[1:384],label=L"\mathrm{TNKF},\;\;R_\mathbf{P}=4 ")
ylims!(0.08, 0.25);
plot!(rootmse_tnsrkf_4,label=L"\mathrm{TNSRKF},\;\;R_\mathbf{L}=4 ")
plot!(vcat(rootmse_tnkf_4[1:379],0.25),label=L"\mathrm{TNKF},\;\;R_\mathbf{P}=16 ")
ylabel!("RMSE",ytickfontsize=20,yguidefontsize=20)
xlabel!(L"t",xtickfontsize=20,xguidefontsize=20)
plot!(legendfontsize=14,legend=:outertop);

plt2 = plot(nll_tnsrkf_1,legend=false)
plot!(nll_tnkf_1)
#plot!(nll_tnsrkf_2)
#plot!(nll_tnkf_2[1:384])
ylims!(-1000, 2000);
plot!(nll_tnsrkf_4)
plot!(nll_tnkf_4)
ylabel!("NLL",ytickfontsize=20,yguidefontsize=20)
xlabel!(L"t",xtickfontsize=20,xguidefontsize=20)

plt = plot(plt1,plt2,layout=(1,2),linewidth=2)
savefig(plt,"./figures/watertanks_RMSE_NLL_RL_1_4.png")

# plot predictions
plt1 = plot(ytest[33:end],label="truth"); plot!(mstar_1[:,100], ribbon=(2σstar_1[:,100],2σstar_1[:,100]),label=L"$\mathbf{m}_*\mid\mathbf{y}_{1:100}$"); title!("(a)");
plot!(ytickfontsize=20,xtickfontsize=20,xguidefontsize=20,linewidth=5);
plot!(legendfontsize=20); 
plt2 = plot(ytest[33:end],label="truth"); plot!(mstar_1[:,200], ribbon=(2σstar_1[:,200],2σstar_1[:,200]), label=L"$\mathbf{m}_*\mid\mathbf{y}_{1:200}$");title!("(b)");
plot!(ytickfontsize=20,xtickfontsize=20,xguidefontsize=20,linewidth=5);
plot!(legendfontsize=20);
plt3 = plot(ytest[33:end],label="truth"); plot!(mstar_1[:,992], ribbon=(2σstar_1[:,992],2σstar_1[:,992]), label=L"$\mathbf{m}_*\mid\mathbf{y}_{1:992}$");title!("(c)");
plot!(ytickfontsize=20,xtickfontsize=20,xguidefontsize=20,linewidth=5);
plot!(legendfontsize=20);

plt = plot(plt1,plt2,plt3,layout=(3,1),size=(500,600))
savefig(plt,"./figures/watertanks_predictions_R_1.png")

plt1 = plot(ytest[33:end],label="truth"); plot!(mstar_4[:,100], ribbon=(2σstar_4[:,100],2σstar_4[:,100]),label=L"$\mathbf{m}_*\mid\mathbf{y}_{1:100}$"); title!("(a)");
plot!(ytickfontsize=20,xtickfontsize=20,xguidefontsize=20,linewidth=5);
plot!(legendfontsize=20);
plt2 = plot(ytest[33:end],label="truth"); plot!(mstar_4[:,200], ribbon=(2σstar_4[:,200],2σstar_4[:,200]), label=L"$\mathbf{m}_*\mid\mathbf{y}_{1:200}$");title!("(b)");
plot!(ytickfontsize=20,xtickfontsize=20,xguidefontsize=20,linewidth=5);
plot!(legendfontsize=20);
plt3 = plot(ytest[33:end],label="truth"); plot!(mstar_4[:,992], ribbon=(2σstar_4[:,992],2σstar_4[:,992]), label=L"$\mathbf{m}_*\mid\mathbf{y}_{1:992}$");title!("(c)");
plot!(ytickfontsize=20,xtickfontsize=20,xguidefontsize=20,linewidth=5);
plot!(legendfontsize=20);

plt = plot(plt1,plt2,plt3,layout=(3,1),size=(500,600))
savefig(plt,"./figures/watertanks_predictions_R_4.png")