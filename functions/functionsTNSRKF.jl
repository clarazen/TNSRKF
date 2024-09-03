module functionsTNSRKF

using ..functionsTT, LinearAlgebra, ..functions_KhatriRao_Kronecker, ..functionsTTmatmul, ..functionsbasicoperations, Random

export TNSRKF, predictions

    function meanupdate(yt,Φt,mtt0,mt,LΦS,Lt,maxiter)
        # the norm core dd is the one in Lt that has the augmented column index

        D       = size(Φt,1)
        dd      = mtt0.normcore
        swipe   = [collect(dd:D)...,collect(D-1:-1:2)...,collect(1:dd-1)...];
        Dir     = Int.([ones(1,D-dd)...,-ones(1,D-1)...,ones(1,dd-1)...]);
        res     = zeros(maxiter,2D-2)

        mtt     = copyTT(mtt0)
        for iter = 1:maxiter
            for k = 1:2D-2
                d           = swipe[k]
                et          = yt - (transpose(khr2ttm(Φt))*mt)[1]
                Wd          = getttm(mtt,d)
                WL          = multiplyTT(transpose(Wd),Lt)
                Wdm         = transpose(Wd)*mt
                WdKe        = WL*LΦS*et

                mtt[d]      = reshape(Wdm + WdKe, size(mtt[d]))
                res[iter,k] = norm(yt - (transpose(khr2ttm(Φt))*mtt)[1])/norm(yt)
                mtt         = shiftTTnorm(mtt,d,Dir[k])
            end
        end

        return mtt,res
    end

    function covupdate(initcov,dd,Φt,priorcov,maxiter,σ²,t,invSt,LΦS)

        Ltt2        = copyTT(initcov)
        Lt          = copyTT(priorcov)

        D           = size(Φt,1)

        Φttm        = transpose(khr2ttm(Φt))
        LΦσS        = copyTT(LΦS)
        LΦσS[dd]    = LΦσS[dd]*sqrt(σ²)
        
        swipe       = [collect(dd:D)...,collect(D-1:-1:2)...,collect(1:dd-1)...];
        Dir         = Int.([ones(1,D-dd)...,-ones(1,D-1)...,ones(1,dd-1)...]);
        sv          = 1
        for iter = 1:maxiter
            for k = 1:2D-2
                d       = swipe[k];

                Ltt_1   = copyTT(Ltt2)
                Ltt_1[d]= Ltt_1[d][:,:,1:size(Ltt_1[d],2),:]
                Qd      = getttm(ttm2ttv(Ltt_1),d)
                Term1   = reshape(transpose(Qd)*ttm2ttv(Lt),size(Ltt_1[d]))
                Term2   = term2(Φttm,Lt,Ltt_1,invSt,d)

                Ltt_2   = copyTT(Ltt2)
                Ltt_2[d]= Ltt_2[d][:,:,size(Ltt_2[d],2)+1:end,:]
                if t == 1
                    Term3   = term3(Ltt_2,priorcov,priorcov,LΦσS,d)
                else
                    Term3   = term3(Ltt_2,initcov,priorcov,LΦσS,d)
                end
                
                Ltt2[d] = cat(Term1-Term2,Term3,dims=3)

                if iter == maxiter && k == 2D-2
                    Lt,Ltt2,sv = propcols(Ltt2,d,Dir[k],true)
                else
                    Ltt2    = propcols(Ltt2,d,Dir[k],false)
                end
            end
        end
        
        return Lt,Ltt2,sv
    end


    function term3(Ltt_2,Ltt2,Lt,LΦσS,d)
        D           = order(Ltt_2)
        Ltt_2_       = Vector{Array{Float64,4}}(undef,D)
        Qtmp        = transpose(getttm(ttm2ttv(transpose(Ltt_2)),d))
        sz          = size(Qtmp[1])
        tmp         = reshape(Qtmp[1],sz[1],sz[2],Int(size(Lt[1],3)),size(Lt[1],2),sz[4])
        Ltt_2_[1]    = reshape(tmp[:,:,1,:,:,:],sz[1],sz[2],size(Lt[1],2),sz[4])
    
        for i = 2:D-1
            sz          = size(Qtmp[i])
            tmp         = reshape(Qtmp[i],sz[1],sz[2],size(Ltt_2[i],3),size(Lt[i],2),sz[4])
            Ltt_2_[i]    = tmp[:,:,1,:,:]
        end
    
        sz          = size(Qtmp[D])
        tmp         = reshape(Qtmp[D],sz[1],sz[2],size(Lt[D],3),size(Lt[D],2),sz[4])
        Ltt_2_[D]    = tmp[:,:,Int(size(Lt[D],3)/2+1),:,:]
    
        Term3   = multiplyTT(TT(Ltt_2_),Ltt2)*LΦσS
        sz      = size(Ltt_2[d])
        Term3   = permutedims(reshape(Term3,sz[1],sz[3],sz[2],sz[4]),[1,3,2,4])
    
        return Term3
    end

    function term2(Φt,Lt,Ltt2,invSt,d)
        D       = order(Lt)
        left    = Vector{Array}(undef,D);
        right   = Vector{Array}(undef,D);
        QT      = transpose(Ltt2);
        LT      = transpose(Lt);
        ΦT      = transpose(Φt);
        ΦTS     = permutedims(tmult(ΦT[1],invSt,[[1,2,4],[3]],[[1],[2]]),[1,2,4,3])

        ## LEFT CORES
        # FIRST CORES
        if d > 1
            ΦL      = tmult(Φt[1], Lt[1],[[2,4],             [1,3]], [[1,2],[3,4]])
            ΦLQ     = tmult(ΦL   , QT[1],[[1,2,4],           [3]],   [[1,2],[3,4]])
            ΦLQL    = tmult(ΦLQ  , Lt[1],[[1,2,3,5],         [4]],   [[1,2],[3,4]])
            ΦLQLL   = tmult(ΦLQL , LT[1],[[1,2,3,4,6],       [5]],   [[1,2],[3,4]])
            ΦLQLLΦS = tmult(ΦLQLL, ΦTS,  [[2,3,4,5,7],       [1,6]], [[1,3,2],[4]])

            left[1] = ΦLQLLΦS 
        end
        # ALL OTHER LEFT CORES
        for i = 2:d-1
            lefti  = copy(left[i-1])
            lefti  = tmult(lefti,Φt[i], [[2,3,4,5,6],[1]],     [[1,2],[3,4]])
            lefti  = tmult(lefti,Lt[i], [[2,3,4,5,7],[1,6]],   [[1,2],[3,4]])
            lefti  = tmult(lefti,QT[i], [[2,3,4,5,7],[1,6]],   [[1,2],[3,4]])
            lefti  = tmult(lefti,Lt[i], [[2,3,4,5,7],[1,6]],   [[1,2],[3,4]])
            lefti  = tmult(lefti,LT[i], [[2,3,4,5,7],[1,6]],   [[1,2],[3,4]])
            lefti  = tmult(lefti,ΦT[i], [[2,3,4,5,7],[1,6]],   [[1,2,3],[4]])

            left[i] = lefti
        end

        ## RIGHT CORES
        # LAST CORES
        if d < D
            ΦL      = tmult(Φt[D], Lt[D],[[1],               [2,3,4]],   [[2,4],[1,3]])
            ΦLQ     = tmult(ΦL   , QT[D],[[1,2],             [3]],       [[2,4],[1,3]])
            ΦLQL    = tmult(ΦLQ  , Lt[D],[[1,2,3],           [4]],       [[2,4],[1,3]])
            ΦLQLL   = tmult(ΦLQL , LT[D],[[1,2,3,4],         [5]],       [[2,4],[1,3]])
            ΦLQLLΦ  = tmult(ΦLQLL, ΦT[D],[[1,2,3,4,5],       [6]],       [[2,3,4],[1]])
            right[D]= ΦLQLLΦ
        end
        # ALL OTHER RIGHT CORES
        for i = D-1:-1:d+1 
            righti  = copy(right[i+1])
            righti  = tmult(Φt[i],righti, [[1,3],[2,4]],     [[1],  [2,3,4,5,6]])
            righti  = tmult(Lt[i],righti, [[1,3],[2,4]],     [[2,3],[1,4,5,6,7]])
            righti  = tmult(QT[i],righti, [[1,3],[2,4]],     [[2,4],[1,3,5,6,7]])
            righti  = tmult(Lt[i],righti, [[1,3],[2,4]],     [[2,5],[1,3,4,6,7]])
            righti  = tmult(LT[i],righti, [[1,3],[2,4]],     [[2,6],[1,3,4,5,7]])
            righti  = tmult(ΦT[i],righti, [[1],[2,3,4]],     [[2,7],[1,3,4,5,6]])
            righti  = permutedims(righti,[6,5,4,3,2,1])

            right[i] = righti
        end

        if d == 1 
            righti  = tmult(Φt[1],right[2],     [[2,3],[1,4]],      [[1],     [2,3,4,5,6]])
            righti  = tmult(Lt[1],righti,       [[3],[1,2,4]],      [[2,3],   [1,4,5,6,7]])
            righti  = tmult(Lt[1],righti,       [[2,3],[1,4]],      [[4],     [1,2,3,5,6]])
            righti  = tmult(LT[1],righti,       [[3],[1,2,4]],      [[2,6],   [1,3,4,5,7]])
            righti  = tmult(ΦTS,righti,         [[1],[2,3,4]],      [[1,4,6], [2,3,5]])
            newcore = permutedims(righti,[1,2,3,4])
        elseif d == D
            lefti = tmult(left[D-1],Φt[D],      [[2,3,4,5,6],[1]],  [[1,2,4],[3]])
            lefti = tmult(lefti,Lt[D],          [[2,3,4,5],[1,6]],  [[1,2,4],[3]])
            lefti = tmult(lefti,Lt[D],          [[1,3,4,5],[2]],    [[1,4],[2,3]])
            lefti = tmult(lefti,LT[D],          [[1,3,4,5],[2,6]],  [[1,2,4],[3]])
            lefti = tmult(lefti,ΦT[D],          [[1,3,4],[2,5]],    [[1,3,2],[4]])
            newcore = permutedims(lefti,[1,3,2,4])
        else
            lefti = tmult(left[d-1],ΦT[d],      [[1,2,3,4,5],[6]],  [[1,3],[2,4]])
            lefti = tmult(lefti,LT[d],          [[1,2,3,4,7],[5,6]],[[1,3],[2,4]])
            lefti = tmult(lefti,Lt[d],          [[1,2,3,5,7],[4,6]],[[1,3],[2,4]])
            lr    = tmult(lefti,right[d+1],     [[1,2,3,6],[7,5,4]],[[4,5,6],[1,2,3]])
            lr    = tmult(lr,  Φt[d],           [[2,3,4,6,7],[1,5]],[[1,2,4],[3]])
            lr    = tmult(lr,  Lt[d],           [[2,3,5],[1,6,4]],  [[1,2,4],[3]])
            newcore = permutedims(lr,[1,2,4,3])
        end

        return newcore
    end

    function propcols(Ltt2,d,dir,last)
        if dir == 1
            sz      = size(Ltt2[d])
            F       = svd(reshape(Ltt2[d],sz[1]*sz[2]*Int(sz[3]/2), 2*sz[4]))
            Ltt2[d] = reshape(F.U[:,1:sz[4]],sz[1],sz[2],Int(sz[3]/2),sz[4])
            SV      = diagm(F.S[1:sz[4]])*F.Vt[1:sz[4],:]
            SV      = reshape(SV,sz[4],2,sz[4])
            sz      = size(Ltt2[d+1])
            tmp     = tmult(SV,Ltt2[d+1],[[1,2],[3]],[[1],[2,3,4]])
            tmp     = permutedims(tmp,[1,3,4,2,5])
            Ltt2[d+1] = reshape(tmp,sz[1],sz[2],sz[3]*2,sz[4])
        elseif dir == -1
            sz      = size(Ltt2[d])
            tmp     = reshape(Ltt2[d],sz[1],sz[2],Int(sz[3]/2),2,sz[4])
            tmp     = permutedims(tmp,[1,4,2,3,5])
            F       = svd(reshape(tmp,sz[1]*2,sz[2]*Int(sz[3]/2)*sz[4])')
            Ltt2[d] = reshape(F.U[:,1:sz[1]]',sz[1],sz[2],Int(sz[3]/2),sz[4])
            VS      = F.V[:,1:sz[1]]*diagm(F.S[1:sz[1]])
            VS      = reshape(VS,sz[1],2,sz[1])
            sz      = size(Ltt2[d-1])
            tmp     = tmult(Ltt2[d-1],VS,[[1,2,3],[4]],[[1],[2,3]])
            Ltt2[d-1] = reshape(tmp,sz[1],sz[2],2*sz[3],sz[4])
        end
        Ltt2.normcore = Ltt2.normcore + dir

        if last == false
            return Ltt2
        else
            # transform Ltt2 ∈ M×2M into M×M
            Ltt     = copyTT(Ltt2)
            sz      = size(Ltt2[d+dir])
            
            # svd of RL I RL x 2J 
            #tmp     = permutedims(Ltt2[d+dir],[1,2,4,3])
            #F       = svd(reshape(tmp,sz[1]*sz[2]*sz[4],sz[3]))
            #tmp     = F.U[:,1:Int(sz[3]/2)]*diagm(F.S[1:Int(sz[3]/2)])
            #tmp     = reshape(tmp,sz[1],sz[2],sz[4],Int(sz[3]/2))
            #Ltt[d+dir]  = permutedims(tmp,[1,2,4,3])
            #sv      =  sum(F.S[Int(sz[3]/2)+1:end]) 

            # svd of RL I J RL x 2
            tmp     = reshape(Ltt2[d+dir],sz[1],sz[2],Int(sz[3]/2),2,sz[4])
            tmp     = permutedims(tmp,[1,2,3,5,4])
            F       = svd(reshape(tmp,sz[1]*sz[2]*Int(sz[3]/2)*sz[4],2))
            tmp     = F.U[:,1]*F.S[1]
            tmp     = reshape(tmp,sz[1],sz[2],sz[4],Int(sz[3]/2))
            Ltt[d+dir]  = permutedims(tmp,[1,2,4,3])
            sv      =  F.S[2]

            return Ltt,Ltt2,sv
        end    

    end

    function getprior(rnks_mean,rnks_cova,M,D,dd,P0)
        # prior mean initialization: random initial TT and zero mean prior
        m0      = initTT(rnks_mean,M,dd,D)
        mtt     = copyTT(m0)
        mt      = TT(m0.cores .* 0)

        # prior covariance root
        factors = sqrt.(diagm.(P0));
        Lt      = factors2ttm(factors);
        sz      = size(Lt)
        
        sz[2,dd]= 2*sz[2,dd]
        Ltt      = ttv2ttm(initTT(rnks_cova,prod.(eachcol(sz)),dd,D),sz);

        return mt,mtt,Lt,Ltt
    end

    function getKalmangain(Φt,Lt,σ²)
        Φttm        = transpose(khr2ttm(Φt))
        ΦL          = multiplyTT(Φttm,Lt)
        invSt       = inv(ΦL*transpose(ΦL) + σ²*I)
        ΦS          = transpose(Φttm)*invSt
        LΦS         = multiplyTT(transpose(Lt),ΦS)

        return LΦS,invSt
    end

    function TNSRKF(N,y,Φ,rnks_mean,rnks_cova,M,dd,P0,maxiter,σ²)
        # computes mean update with ALS and covariance root with ALS
        
        D           = size(Φ,1)

        # compute prior mean and covariance in TT format as well as TT initialization
        mt,mtt,priorcov,initcov = getprior(rnks_mean,rnks_cova,M,D,dd,P0)

        # recursion
        resall      = zeros(N,(2D-2)*maxiter)
        normLt_all  = zeros(N,(2D-2)*maxiter)
        mt_all      = Vector{TTv}(undef,N)
        Lt_all      = Vector{TTm}(undef,N)
        Lt2_all     = Vector{TTm}(undef,N)

        sv          = zeros(N)
        println("time-step: ")
        for t = 1:5#size(Φ[1],1)
            if mod(t, 20) == 0
                print(t,",")
            end
            Φt          = getΦentries(Φ,t:t)

            # update mean with ALS
            mtt0        = copyTT(mtt)

            # Compute Kalman gain using priorcov (MxM) at t=1 and initcov (Mx2M) at t>1           
            if t == 1
                LΦS,invSt   = getKalmangain(Φt,priorcov,σ²)
                mtt,res     = meanupdate(y[t],Φt,mtt0,mt,LΦS,priorcov,maxiter)
            else
                LΦS,invSt   = getKalmangain(Φt,initcov,σ²)
                mtt,res     = meanupdate(y[t],Φt,mtt0,mt,LΦS,initcov,maxiter)
            end

            # update prior for next time step / measurement
            overwriteTT!(mt,mtt)

            # update covariance
            Lt,Ltt2,sv[t] = covupdate(initcov,dd,Φt,priorcov,maxiter,σ²,t,invSt,LΦS);

            mt_all[t]  = mtt
            Lt_all[t]  = Lt
            Lt2_all[t] = Ltt2

            # for next iteration
            initcov     = copyTT(Ltt2);
            priorcov    = copyTT(Lt);

            resall[t,:]   = res[:]
            #normLt_all[t,:] = normLt[:]
        end
        println("done")

        return mt_all,Lt_all,Lt2_all,sv#,resall,normLt_all,sv
    end

    function predictions(N,Nstar,Φs,mt_all,Lt_all,ytest)
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
                ΦsiL            = multiplyTT(transpose(khr2ttm(Φsi)),Lt_all[t])
                σstar[nstar,t]  = ΦsiL*transpose(ΦsiL)
            end
        end
        println("done")

        rootmse = [sqrt(sum((mstar[:,t] - ytest).^2/Nstar)) for t=1:N]
        nll     = [sum(.5*(log.(2π*σstar[:,t]) + (ytest - mstar[:,t]).^2 ./ σstar[:,t])) for t=1:N]

        return rootmse,nll
    end

end