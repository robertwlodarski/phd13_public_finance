# Main functions 
# 1. Last period's utilities 
# 2. Last period's policies 
# 3. Residual function 
# 4. Backward loop induction
# 5. Run the model 

# 1. Last period's utilities 
function fnUtilityLast(params)

    # A. Unpacking business 
    @unpack γ, χ, σ, a⃗, y⃗, L, h, c̲ = params 

    # B. Working utility 
    𝐕ʷ              = ((max.(a⃗' .+ y⃗,c̲)).^(1-γ) - 1) ./ (1-γ) .- χ * ((L - h)^(1 - σ) - 1)/(1 - σ)
    
    # C. Non-working utility 
    𝐕ⁿʷ             = ((max.(a⃗,c̲)).^(1-γ) - 1) ./ (1-γ)
    
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
    endo.𝐂[T,:,:]       .= a⃗ .+ (y⃗ .* endo.𝐍[T,:,:])'
    endo.𝐀[T,:,:]       .= zeros(Nᵃ, Nʸ)

end 

# 3. Find assets
function fnFindAssets(iy, ia,it,RHS_spline, params, endo)

    # A. Unpack parameters 
    @unpack a⃗,y⃗,γ,r, c̲, β, a̲= params

    # B. Check the lower bound constraint (working)
    Lower   = a̲ 
    Upper   = (1+r) * (a⃗[ia] + y⃗[iy] - c̲)
    Res_L   = (a⃗[ia] + y⃗[iy] - Lower/(1+r))^(-γ) - β * (1+r) * endo.𝔼∂𝐕[it, iy, 1]
    Constr  = (Res_L >= 0)

    # C. Spline the strictly positive RHS (working)
    if Constr == false 
        A_w     = find_zero(a -> (a⃗[ia] + y⃗[iy] - a/(1+r))^(-γ) - β * (1+r) * RHS_spline(a), (Lower, Upper), Bisection())
    else 
        A_w     = Lower
    end

    # D. Check the lower bound constraint (working)
    Cash_nw     = a⃗[ia]
    if Cash_nw <=  c̲ + a̲ / (1 + r)
        A_nw    = a̲
    else
        Lower_n     = a̲ 
        Upper_n     = (1+r) * (a⃗[ia] - c̲)
        Res_Ln      = (a⃗[ia] - Lower_n/(1+r))^(-γ) - β * (1+r) * endo.𝔼∂𝐕[it, iy, 1]
        Constr_n    = (Res_Ln >= 0)

        # E. Spline the strictly positive RHS (working)
        if Constr_n == false 
            A_nw    = find_zero(a -> (a⃗[ia] - a/(1+r))^(-γ) - β * (1+r) * RHS_spline(a), (Lower_n, Upper_n), Bisection())
        else 
            A_nw    = Lower_n
        end 
    end 
    return A_w, A_nw
end 

# Backward loop induction 

function fnBackwardInduction!(params, endo; end_labour = end_labour)

    # A. Unpacking business 
    @unpack T,a̲,a⃗,Γ,γ,Nʸ,Nᵃ,y⃗,r,χ,σ,L,h,β = params

    # B. Get last period's policies 
    fnLastPeriod!(params, endo; end_labour = true)

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