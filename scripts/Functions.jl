# Main functions 
# 1. Last period's utilities 
# 2. Last period's policies 
# 3. Residual function 
# 4. Backward loop induction
# 5. Non-stochastic simulation 
# 6. Stochastic simulation

# 1. Last period's utilities 
function fnUtilityLast(params)

    # A. Unpacking business 
    @unpack γ, χ, σ, a⃗, y⃗, L, h, c̲ = params 

    # B. Working utility 
    𝐕ʷ              = ((max.(a⃗' .+ y⃗,c̲)).^(1-γ) .- 1) ./ (1-γ) .- χ * ((L - h)^(1 - σ) - 1)/(1 - σ)
    
    # C. Non-working utility 
    𝐕ⁿʷ             = ((max.(a⃗',c̲)).^(1-γ) .- 1) ./ (1-γ)
    
    return 𝐕ʷ, 𝐕ⁿʷ
end 

# 2. Last period's policies 
function fnLastPeriod!(params, endo; end_labour = end_labour)

    # A. Unpacking business 
    @unpack T, a⃗, y⃗, Nᵃ, Nʸ, Γ, γ = params

    # B. Compute values with and without working 
    𝐕ʷ, 𝐕ⁿʷ                 = fnUtilityLast(params)
    if end_labour == true
        endo.𝐍[T,:,:]       .= (𝐕ʷ .>= 𝐕ⁿʷ)
        endo.𝐕[T,:,:]       .= max.(𝐕ʷ, 𝐕ⁿʷ)
    else 
        endo.𝐍[T,:,:]       .= true 
        endo.𝐕[T,:,:]       .= 𝐕ʷ
    end 

    # B. Choose consumption and savings 
    # Given that V(T+1)= 0 ∀ Aₜ₊₁, agents consume everything.
    # Cₜ=Aₜ+Yₜ for t = T
    endo.𝐂[T,:,:]       .= a⃗' .+ (y⃗ .* endo.𝐍[T,:,:])
    endo.𝐀[T,:,:]       .= zeros(Nʸ,Nᵃ)

end 

# 3. Find assets
function fnFindAssets(iy, ia, it, RHS_spline, params, endo)

    # A. Unpack parameters 
    @unpack a⃗, y⃗, γ, r, c̲, β, a̲ = params

    # B. Helper function: The Euler residual 
    function Res(a_next, cash)
        return (cash - a_next/(1+r))^(-γ) - β * (1+r) * RHS_spline(a_next)
    end

    # C. Working state
    Cash_w = a⃗[ia] + y⃗[iy]
    if Cash_w <= c̲ + a̲ / (1+r)
        A_w = a̲
    else
        Lower_w = a̲ 
        Upper_w = (1+r) * (Cash_w - c̲)
        
        # Explicitly check both bounds to prevent Bisection crashes
        if Res(Lower_w, Cash_w) >= 0.0
            A_w = Lower_w
        elseif Res(Upper_w, Cash_w) <= 0.0
            A_w = Upper_w
        else 
            A_w = find_zero(a -> Res(a, Cash_w), (Lower_w, Upper_w), Bisection())
        end
    end

    # D. --- NON-WORKING STATE ---
    Cash_nw = a⃗[ia]
    if Cash_nw <= c̲ + a̲ / (1+r)
        # Starvation trap prevention
        A_nw = a̲
    else
        Lower_n = a̲ 
        Upper_n = (1+r) * (Cash_nw - c̲)
        
        # Explicitly check both bounds to prevent Bisection crashes
        if Res(Lower_n, Cash_nw) >= 0.0
            A_nw = Lower_n
        elseif Res(Upper_n, Cash_nw) <= 0.0
            A_nw = Upper_n
        else 
            A_nw = find_zero(a -> Res(a, Cash_nw), (Lower_n, Upper_n), Bisection())
        end 
    end 

    return A_w, A_nw
end

# Backward loop induction 

function fnBackwardInduction!(params, endo; end_labour = end_labour)

    # A. Unpacking business 
    @unpack T,a̲,a⃗,Γ,γ,Nʸ,Nᵃ,y⃗,r,χ,σ,L,h,β = params

    # B. Get last period's policies 
    fnLastPeriod!(params, endo; end_labour = end_labour)

    # C. Start the loop 
    for it in T-1:(-1):1

        # I. Update the expected values 
        endo.𝔼𝐕[it,:,:]             .= Γ * (@views endo.𝐕[it+1,:,:])
        endo.𝔼∂𝐕[it,:,:]            .= Γ * ((@views endo.𝐂[it+1,:,:])).^(-γ)

        for iy in 1:1:Nʸ
            ℑᶠ                      = Spline1D(a⃗,(@views endo.𝔼∂𝐕[it, iy, :]); k=1, bc="extrapolate")
            ℑᵛ                      = Spline1D(a⃗,(@views endo.𝔼𝐕[it,iy,:]); k=1, bc="extrapolate")
            for ia in 1:1:Nᵃ
                
                if end_labour == true 
                    # Model with endogenous labour supply 
                    # II. Asset policy functions
                    Aʷ, Aⁿʷ             = fnFindAssets(iy, ia,it,ℑᶠ, params, endo)

                    # III. Consumption policies
                    Cʷ                  = y⃗[iy] + a⃗[ia] - Aʷ/(1+r)
                    Cⁿʷ                 = a⃗[ia] - Aⁿʷ/(1+r)

                    # IV. Value functions 
                    Vʷ                  = ((Cʷ)^(1-γ)-1)/(1-γ) - χ * ((L-h)^(1-σ)-1)/(1-σ) + β * ℑᵛ(Aʷ)
                    Vⁿʷ                 = ((Cⁿʷ)^(1-γ)-1)/(1-γ)+ β * ℑᵛ(Aⁿʷ)

                    # V. Policies 
                    endo.𝐍[it,iy,ia]    = (Vʷ >= Vⁿʷ)
                    endo.𝐕[it,iy,ia]    = Vⁿʷ + endo.𝐍[it,iy,ia]*(Vʷ - Vⁿʷ)
                    endo.𝐂[it,iy,ia]    = Cⁿʷ + endo.𝐍[it,iy,ia]*(Cʷ - Cⁿʷ)
                    endo.𝐀[it,iy,ia]    = Aⁿʷ + endo.𝐍[it,iy,ia]*(Aʷ - Aⁿʷ)
                
                else 
                    # VI. Version without endogeus labour supply
                    A,_                 =  fnFindAssets(iy, ia,it,ℑᶠ, params, endo)
                    endo.𝐀[it,iy,ia]    = A
                    endo.𝐂[it,iy,ia]    = y⃗[iy] + a⃗[ia] - A/(1+r)
                    endo.𝐕[it,iy,ia]    = ((endo.𝐂[it,iy,ia])^(1-γ)-1)/(1-γ)+ β * ℑᵛ(endo.𝐀[it,iy,ia])
                end 
            end 
        end 
    end 
end 

# 5. Non-stochastic simultation (Young, 2010)
function fnNonStochasticSimulation!(params,endo; end_labour = true)

    # A. Unpacking business 
    @unpack T,y⃗,a⃗,Γ = params 
    # B. Run the model 
    fnBackwardInduction!(params, endo; end_labour = end_labour)

    # C. Start iterating 
    for it in 1:1:T-1 
        for iy in eachindex(y⃗)
            for ia in eachindex(a⃗)
                Mass        = endo.Φ[it,iy,ia]
                if Mass == 0.0
                    continue 
                end
                Aₜ          = endo.𝐀[it,iy,ia]
                ib          = clamp(searchsortedlast(a⃗,Aₜ),1,length(a⃗)-1)
                iu          = ib + 1
                w           = clamp((a⃗[iu]-Aₜ) / (a⃗[iu]-a⃗[ib]),0.0,1.0)

                # D. Allocate mass to each potential shock value 
                for iyp in eachindex(y⃗)
                    endo.Φ[it+1,iyp,iu]     += Mass * Γ[iy,iyp] * (1-w)
                    endo.Φ[it+1,iyp,ib]     += Mass * Γ[iy,iyp] * w
                end 
            end 
        end 
        
        # D, Update the marginal wealth distribution for tomorrow
        for ia in eachindex(a⃗)
                endo.𝔼ʸΦ[it+1, ia] = sum(@views endo.Φ[it+1, :, ia])
        end
    end 
end 

# 6. Stochastic simulation
function fnMonteCarlo!(params, endo; end_labour = true)

    # A. Unpacking business 
    @unpack T, y⃗,a⃗,Γ, S,sʳⁿᵍ,ν⃗ = params 

    # B. Sampling indices for the initial income process 
    rng             = Xoshiro(sʳⁿᵍ)
    Yᵢ              = zeros(Int, T,S)
    Yᵢ[1,:]         = sample(rng, 1:length(y⃗), Weights(ν⃗),S)

    # C. Start the life-cycle loop
    for it in 1:T-1
        
        # I. Build splines for policy functions 
        SplineA     = [Spline1D(a⃗, @views(endo.𝐀[it,iy,:]); k=1, bc = "extrapolate") for iy in eachindex(y⃗)]
        SplineC     = [Spline1D(a⃗, @views(endo.𝐂[it,iy,:]); k=1, bc = "extrapolate") for iy in eachindex(y⃗)]
        if end_labour == true 
            SplineN = [Spline1D(a⃗, Float64.(@views(endo.𝐍[it,iy,:])); k=1, bc = "extrapolate") for iy in eachindex(y⃗)]
        end 

        # II. Loop for all S agents 
        for is in 1:S
            
            # IIA. Values 
            iy              = Yᵢ[it,is]
            a               = endo.Â[it,is]

            # IIB. Continuous choices 
            endo.Â[it+1,is] = SplineA[iy](a)
            endo.Ĉ[it,is]   = SplineC[iy](a)
            endo.Ŷ[it,is]   = y⃗[iy]

            # IIC. Discrete choices 
            if end_labour == true
                UsedSpline      = clamp(SplineN[iy](a),0.0,1.0)
                endo.N̂[it,is]   = round(Bool,UsedSpline)
            end 

            # IID. Tomorrow's income shock 
            Yᵢ[it+1,is]         = sample(rng, 1:length(y⃗), Weights(@views(Γ[iy,:])))
        end 
    end

    # D. Last period 
    if end_labour == true 
            SplineN = [Spline1D(a⃗, Float64.(@views(endo.𝐍[T,iy,:])); k=1, bc = "extrapolate") for iy in eachindex(y⃗)]
        end 
    for is in 1:S 

        # I. Assets and income shock index 
        a       = endo.Â[T,is]
        iy      = Yᵢ[T,is]

        # II. Discrete choices 
        if end_labour == true
            UsedSpline      = clamp(SplineN[iy](a),0.0,1.0)
            endo.N̂[T,is]   = round(Bool,UsedSpline)
        end  

        # III. Eat everything in the last period
        endo.Ĉ[T,is]    = a + endo.N̂[T,is] * y⃗[iy]
        endo.Ŷ[T,is]    = y⃗[iy]
    end 
end 

