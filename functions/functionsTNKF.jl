module functionsTNKF
# Based on code by Batslier et al.: A Tensor Network Kalman filter with an application in recursive MIMO Volterra system identification

using ..functionsTT, ..functionsbasicoperations, ..functions_KhatriRao_Kronecker, LinearAlgebra

export loopTNKF, loopKF, predictions_TNKF

    function loopTNKF(y,Φ,N,M,D,P0,rnks_mean,rnks_cova,dd,σ²)
        # init
        m0              = initTT(rnks_mean,M,dd,D);
        mt              = TT(m0.cores .* 0);
        factors         = diagm.(P0);
        Pt              = factors2ttm(factors);

        # loop 
        mt_all          = Vector{TTv}(undef,N)
        Pt_all          = Vector{TTm}(undef,N)
        print("time-step: ")
        for t = 1:N
            if mod(t, 20) == 0
                print(t,",")
            end
            yt      = y[t]
            Φt      = getΦentries(Φ,t:t);
            mtt,Ptt = TNKF(mt,Pt,Φt,yt,σ²,rnks_mean,rnks_cova)
            mt_all[t] = mtt
            Pt_all[t] = Ptt
            overwriteTT!(mt,mtt)
            overwriteTT!(Pt,Ptt)
        end
        print("done")
        return mt_all,Pt_all
    end


    function TNKF(mt,Pt,Φt,yt,σ²,rnks_mean,rnks_cova)
        
        Φttm            = khr2ttm(Φt);
        
        ΦPΦ             = ttm2mat(multiplyTT(transpose(Φttm),multiplyTT(Pt,Φttm)))[1];
        St              = ΦPΦ + σ²
        invSt           = inv(St);
        Kt              = multiplyTT(Pt,Φttm)*invSt;
        
        Δm              = ttm2ttv(Kt*(yt - (transpose(Φttm)*mt)[1]))
        
        mtt             = mt + Δm;

        mtt             = roundTT(mtt,rnks_mean);


        Ptt             = Pt - multiplyTT(Kt*St,transpose(Kt))
        Ptt             = roundTT(Ptt,rnks_cova)

        return mtt,Ptt

    end

    function loopKF(M,N,Nstar,Φ,Φs,y,σ²,P0)
        # init
        D               = length(M)
        mt              = zeros(prod(M))
        factors         = diagm.(P0);
        Pt              = ttm2mat(factors2ttm(factors))
        mt_all          = zeros(N,size(mt,1))
        Pt_all          = zeros(N,size(Pt)...)
        mstar           = zeros(Nstar,N);
        σstar           = zeros(Nstar,N);

        println("time-step: ")
        for t = 1:N
            if mod(t, 20) == 0
                print(t,",")
            end
            yt              = y[t]
            Φt              = (ttm2mat(factors2ttm(getΦentries(Φ,t:t))))[1,:]
            St              = Φt'*Pt*Φt + σ²
            invSt           = inv(St);
            Kt              = Pt*Φt*invSt;
            
            mt              = (mt + Kt*(yt - (Φt'*mt)[1]))[:,1]
            Pt              = Pt - Kt*St*Kt'
            mt_all[t,:]     = mt
            Pt_all[t,:,:]   = Pt
    
            for nstar = 1:Nstar
                Φsi             = (ttm2mat(factors2ttm(getΦentries(Φs,nstar:nstar))))[1,:]
                mstar[nstar,t]  = (Φsi'*mt)[1]
                ΦsiPΦsi         = Φsi'*Pt*Φsi
                σstar[nstar,t]  = sqrt.(ΦsiPΦsi)[1]
            end
    
        end
        print("done")
        return mstar,σstar,mt_all,Pt_all
    end

    function predictions_TNKF(N,Nstar,Φs,mt_all,Pt_all,ytest,Xtest)
        mstar = zeros(Nstar,N);
        σstar = zeros(Nstar,N);
        println("test point: ")
        for nstar = 1:Nstar
            if mod(nstar, 20) == 0
                print(nstar,",")
            end
            Φsi     = getΦentries(Φs,nstar:nstar)
            for t = 1:N
                mstar[nstar,t]  = (transpose(khr2ttm(Φsi))*mt_all[t])[1]
                σstar[nstar,t]  = (multiplyTT(transpose(khr2ttm(Φsi)),Pt_all[t])*khr2ttm(Φsi))[1]
                if σstar[nstar,t] < 0
                    σstar[nstar,t] = 0
                end
            end
        end
        print("done")

        rootmse = [sqrt(sum((mstar[:,t] - ytest).^2/size(Xtest,1))) for t=1:N]
        nll     = [sum(.5*(log.(2π*σstar[:,t]) + (ytest - mstar[:,t]).^2 ./ σstar[:,t])) for t=1:N]

        return rootmse,nll
    end

end
