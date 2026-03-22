# Content 
# 1. Parameters (structure and constructor)
# 2. Endogenous variables (structure and constructor)

# 1. Parameters (structure)
@with_kw struct ModelParameters

    # A. Parameters 
    γ::Float64          = 4.0           # Consumption utility parameter 
    χ::Float64          = 0.5           # Disutility of labour shifter
    σ::Float64          = 4.0           # Labour utility parameter 
    β::Float64          = 0.98          # Discount factor
    L::Float64          = 1.0           # Leisure endowment (normalised to 1)
    h::Float64          = 8.0/24.0      # Hours if working 
    r::Float64          = 1/β - 1       # Interest rate

    # B. Lifecycle parameters 
    T::Int              = 40            # Years in the labour market 
    c̲::Float64          = 1e-4          # Consumption floor 

    # C. Income grid 
    ρ::Float64          = 0.98          # Persistence
    σʸ::Float64         = sqrt(0.0289)  # Shock standard deviation 
    Nʸ::Int             = 20            # Income grid size 
    y⃗::Vector{Float64}  = zeros(Nʸ)     # Income grid 
    Γ::Matrix{Float64}  = zeros(Nʸ,Nʸ)  # Income transition PDF
    ν⃗::Vector{Float64}  = zeros(Nʸ)     # Stationary distribution of income

    # D. Assets grid 
    Nᵃ::Int             = 80            # Assets grid size 
    a⃗::Vector{Float64}  = zeros(Nᵃ)     # Assets grid 
    a̲::Float64          = 0.0           # Minimum assets 
    a̅::Float64          = 80.0          # Maximum assets
    θᵃ::Float64         = 3.0           # Curvature of the assets grid 
end 

# 2. Parameters (constructor)
function fnSetUpParameters(params::ModelParameters = ModelParameters())

    # A. Unpacking business 
    @unpack ρ, Nʸ, σʸ, Nᵃ, a̲, a̅, θᵃ = params

    # B. Idiosyncratic income items → Rouwenhorst as ρ ≃ 1
    ℳ𝒞                  = rouwenhorst(Nʸ,ρ,σʸ)
    y⃗̃                   = exp.(ℳ𝒞.state_values)             # Non-normalised grid
    ν⃗                   = stationary_distributions(ℳ𝒞)[1]   # Stationary distribution 
    𝔼y⃗̃                  = dot(y⃗̃,ν⃗)                          # Mean of the non-normalised grid

    # D. Save results 
    return reconstruct(params;
        ν⃗               = ν⃗,
        y⃗               = y⃗̃ ./ 𝔼y⃗̃,
        Γ               = ℳ𝒞.p,
        a⃗               = a̲ .+ (a̅ .- a̲) .* (range(0,1,length=Nᵃ)).^θᵃ
    )
end 
UsedParameters = fnSetUpParameters()

# 2. Endogenous variables preallocation (structure)
@with_kw mutable struct EndogenousVariables

    # A. Key value functions 
    𝐕::Array{Float64,3}     # Value function 
    𝔼𝐕::Array{Float64,3}    # Expected value function 
    ∂𝐕::Array{Float64,3}    # Partial derivative of t+1 assets 
    𝔼∂𝐕::Array{Float64,3}   # Derivative of the above 
    𝐀::Array{Float64,3}     # Savings policy function 
    𝐂::Array{Float64,3}     # Consumption policy function 
    𝐍::Array{Bool,3}        # Labour supply policy
end

# 2. Endogenous variables preallocation (constructor)
function fnSetUpEndo(params::ModelParameters)

    # A. Unpacking business 
    @unpack T, Nʸ, Nᵃ = params 

    # B. Preallocate values: Values and policies 
    𝐕       = zeros(T, Nʸ, Nᵃ)
    𝔼𝐕      = zeros(T, Nʸ, Nᵃ)
    ∂𝐕      = zeros(T, Nʸ, Nᵃ)
    𝔼∂𝐕     = zeros(T, Nʸ, Nᵃ)
    𝐀       = zeros(T, Nʸ, Nᵃ)
    𝐂       = zeros(T, Nʸ, Nᵃ)
    𝐍       = fill(true, T, Nʸ, Nᵃ)

    # D. Return 
    return EndogenousVariables(
        𝐕   = 𝐕,
        𝔼𝐕  = 𝔼𝐕,
        ∂𝐕  = ∂𝐕,
        𝔼∂𝐕 = 𝔼∂𝐕,
        𝐀   = 𝐀,
        𝐂   = 𝐂,
        𝐍   = 𝐍
    )   
end 
EndoInelasticLab    = fnSetUpEndo(UsedParameters) # Structure for model with ileastic labour supply 
EndoMain            = fnSetUpEndo(UsedParameters) # Structure for model with elastic labour supply 