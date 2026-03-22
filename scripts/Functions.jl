# Main functions 
# 1. Last period's utilities 
# 2. Last period's policies 
# 3. Residual function 

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
function fnLastPeriod!(params, endo; end_labour = true)

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

    # B. Check the lower bound constraint
    Lower   = a̲ 
    Upper   = (1+r) * (a⃗[ia] + y⃗[iy] - c̲)
    Res_L   = (a⃗[ia] + y⃗[iy] - Lower/(1+r))^(-γ) - β * (1+r) * endo.𝔼∂𝐕[it, iy, 1]
    Constr  = (Res_L >= 0)

    # C. Spline the strictly positive RHS
    if Constr == false 
        A_opt = find_zero(a -> (a⃗[ia] + y⃗[iy] - a/(1+r))^(-γ) - β * (1+r) * RHS_spline(a), (Lower, Upper), Bisection())
    else 
        A_opt = a⃗[1]
    end
    return A_opt
end 

# Backward loop induction 

function fnBackwardInduction!(params, endo; end_labour = true)

    # A. Unpacking business 
    @unpack T,a̲,a⃗,Γ,γ,N,Nᵃ = params

    # B. Get last period's policies 
    fnLastPeriod!(params, endo; end_labour = true)

    # C. Start the loop 
    for it in T-1:(-1):1

        # I. Update the expected values 
        endo.𝔼𝐕[it,:,:]             .= Γ * endo.𝐕[it+1,:,:]
        endo.𝔼∂𝐕[it,:,:]            .= Γ * (endo.𝐂[it+1,:,:]).^(-γ)

        for iy in 1:1:Nʸ
            ℑᶠ                      = Spline1D(a⃗, endo.𝔼∂𝐕[it, iy, :]; k=1, bc="extrapolate")
            for ia in 1:1:Nᵃ
                endo.𝐀[it,iy,ia]    = fnFindAssets(iy, ia,it,ℑᶠ, params, endo)
                
            end 
        end 
    end 
end 