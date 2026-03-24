# Content:
# 1. Synthetic sample generation
# 2. Plots of the key variables 

## 1. Generate synthetic sample 
function fnGenerateSyntheticPanel(params, endo)
    
    # A. Unpacking business
    @unpack T, S,sʳⁿᵍ,σ̃ᵃ,σ̃ʸ,σ̃ᶜ  = params
    rng                 = Xoshiro(sʳⁿᵍ)
    
    # B. Build panel indices 
    age_col     = repeat(1:T, outer = S)
    agent_col   = repeat(1:S, inner = T)
    
    # C. Extract True Continuous Variables
    true_A      = vec(endo.Â)
    true_C      = vec(endo.Ĉ)
    true_Y      = vec(endo.Ŷ)
    work_N      = vec(endo.N̂)
    
    # D. Apply measurement error 
    # I. Assets: Additive noise (allows negative reported wealth, standard for classical ME)
    noise_A     = rand(rng, Normal(0, σ̃ᵃ), T * S)
    obs_A       = true_A .+ noise_A
    
    # II. Consumption & income: Log-additive noise (preserves strictly positive values)
    noise_C     = rand(rng, Normal(0, σ̃ᶜ), T * S)
    noise_Y     = rand(rng, Normal(0,σ̃ʸ), T * S)
    obs_C       = true_C .* exp.(noise_C)
    obs_Y       = true_Y .* exp.(noise_Y)
    
    # E. Assemble the DataFrame
    df_panel = DataFrame(
        AgentID  = agent_col,
        Age      = age_col,
        True_A   = true_A,
        Obs_A    = obs_A,
        True_C   = true_C,
        Obs_C    = obs_C,
        True_Y   = true_Y,
        Obs_Y    = obs_Y,
        Employed = Int.(work_N) 
    )
    return df_panel
end

## 3. Create and save the dataset
SyntheticPanel = fnGenerateSyntheticPanel(UsedParameters, EndoMain)
first(SyntheticPanel, 10)