# Content 
# 1. Parameters (structure and constructor)
# 2. Endogenous variables (structure and constructor)

# 1. Parameters (structure)
@with_kw struct ModelParameters

    # A. Parameters 
    α::Float64          = 0.57  # Labour share 
    γ::Float64          = 0.28  # Capital share  
    β::Float64          = 0.96  # Discount 
    δ::Float64          = 0.08  # Depreciation 
    f::Float64          = 1     # Fixed cost 
    λ::Float64          = 0.1   # Exogenous exit rate 

    # B.Productivity & wedges
    ρ::Float64          = 0.9           # Persistence (log) 
    σ::Float64          = 0.2           # Standard deviation (log)
    φ̄::Float64          = exp(1.39)     # Average productivity 
    Nᵩ::Int             = 100           # Productivity grid size 
    φ⃗::Vector{Float64}  = zeros(Nᵩ)     # Productivity grid 
    Nₜ::Int             = 15            # (Uncorrelated) productivity wedge grid
    τ⃗::Vector{Float64}  = zeros(Nₜ)     # Productivity wedge grid
    τ̄::Float64          = 0.2           # Maximum distortion
    τ̲::Float64          = 0.0           # Minimum distortion 
    ξ::Float64          = 0.5           # Correlation parameter b/n productivity and wedges
    g::Matrix64{Float64}= zeros(Nᵩ,Nₜ)  # PDF (productivity and wedge)
    G::Matrix{Float64}  = zeros(Nᵩ,Nₜ)  # CDF (productivity and wedge)
end 

# 2. Parameters (constructor)
function fnSetUpParameters(params::ModelParameters = ModelParameters())

    # A. Unpacking business 
    @unpack φ̄, ρ, σ, Nᵩ, Nₜ, τ̄, τ̲, ξ = params

    # B. Idiosyncratic productivity items → Rouwenhorst as ρ ≃ 1
    # xₜ = ρ xₜ₋₁ + ϵₜ, where xₜ = log φₜ - log φ̄
    ℳ𝒞                  = rouwenhorst(Nᵩ,ρ,σ)               
    φ⃗                   = exp.(ℳ𝒞.state_values .+ log(φ̄))
    τ⃗                   = collect(range(τ̲, τ̄, length=Nₜ))

    # C. Joint distribution of productivity and distortions
    # Create standardised z-score grids (evaluating from -3 to +3 std devs)
    # [This could be improved; to be done later.]
    x_std       = range(-3.0, 3.0, length=Nᵩ)
    y_std       = range(-3.0, 3.0, length=Nₜ)
    Σ           = [1.0 ξ; ξ 1.0]
    dist        = MvNormal([0.0, 0.0], Σ)
    g           = [pdf(dist, [x, y]) for x in x_std, y in y_std]
    g           .= Π_joint ./ sum(Π_joint) # Normalise
    G           = cumsum(cumsum(Π_joint, dims=1), dims=2)

    # D. Save results 
    return reconstruct(params;
        φ⃗   = φ⃗,
        τ⃗   = τ⃗,
        G   = G,
        g   = g
    )
end 
UsedParameters = fnSetUpParameters()

# 2. Endogenous variables preallocation (structure)
@with_kw mutable struct EndogenousVariables

    # A. Key values - precomputed 
    𝐤::Matrix{Float64}      # Matrix of optimal capital chosen
    𝐧::Matrix{Float64}      # Matrix of optimal labour chosen 
    Π::Matrix{Float64}      # Matrix of optimal profit level 
    𝐞::Matrix{Bool}         # Matrix of entry decisions 

    # B. Equilibrium objects to be computed 
    μ::Matrix{Float64}      # Stationary distribution 
    Y::Float64              # Aggregate output 
    Kᴰ::Float64             # Aggregate capital demand
    Nᴰ::Float64             # Aggregate labour demand 
    Kˢ::Float64             # Aggregate capital supply 
end

# 2. Endogenous variables preallocation (constructor)
function fnSetUpEndo(params::ModelParameters)

    # A. Unpacking business 
    @unpack Nᵩ, Nₙ = params 

    # B. Preallocate values: Values and policies 
    𝐤       = zeros(Nᵩ,Nₙ)
    𝐧       = zeros(Nᵩ,Nₙ)
    Π       = zeros(Nᵩ,Nₙ)
    𝐞       = fill(true,Nᵩ,Nₙ)

    # C. Others 
    μ       = zeros(Nᵩ,Nₙ)
    Y       = 0.0
    Kᴰ      = 0.0
    Nᴰ      = 0.0
    Kˢ      = 0.0

    # D. Return 
    return EndogenousVariables(
        𝐤   = 𝐤,
        𝐧   = 𝐧,
        Π   = Π,
        𝐞   = 𝐞,
        μ   = μ,
        Y   = Y,
        Kᴰ  = Kᴰ,
        Nᴰ  = Nᴰ,
        Kˢ  = Kˢ
    )   
end 
Endo    = fnSetUpEndo(UsedParameters)