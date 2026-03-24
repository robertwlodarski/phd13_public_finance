# Content 
# 1. Functions for preparing data
# 2. GMM regressions
# 3. Improvement 

# 1. Functions for preparing data
function fnPrepareRegressionData(df)
    # A. Sort panel
    sort!(df, [:AgentID, :Age])

    # B. Build C_lag and N_lag safely via join
    lag_df          = select(df, :AgentID, :Age, :Obs_C, :Employed)
    lag_df.Age      = lag_df.Age .+ 1
    df              = leftjoin(df, rename(lag_df,
                        :Obs_C    => :C_lag,
                        :Employed => :N_lag), on = [:AgentID, :Age])

    # C. Consumption growth and its square — created here first
    df.g            = (df.Obs_C .- df.C_lag) ./ df.C_lag
    df.g_sq         = df.g .^ 2

    # D. Now build g_sq_lag safely via join
    gsq_df          = select(df, :AgentID, :Age, :g_sq)
    gsq_df.Age      = gsq_df.Age .+ 1
    df              = leftjoin(df, rename(gsq_df, :g_sq => :g_sq_lag), on = [:AgentID, :Age])

    # E. Filter to working-working transitions only
    df_euler = filter(row -> coalesce(row.Employed == 1 && row.N_lag == 1, false), df)

    # F. Age variance instrument built on the filtered sample
    age_vars        = combine(groupby(df_euler, :Age), :g_sq => mean => :Age_Var_g)
    df_euler        = leftjoin(df_euler, age_vars, on = :Age)

    # G. Lagged age variance via join
    av_lag          = select(age_vars, :Age, :Age_Var_g)
    av_lag.Age      = av_lag.Age .+ 1
    df_euler        = leftjoin(df_euler, rename(av_lag, :Age_Var_g => :Lag_Age_Var_g), on = :Age)

    return dropmissing(df_euler)
end
df_reg = fnPrepareRegressionData(SyntheticPanel)
combine(groupby(df_reg, :Age), nrow)

# 2. Running GMM regressions 
# A. Helper to extract gamma from the coefficient: γ = 2 * β - 1
function extract_gamma(model)
    beta    = coef(model)[1] 
    return (2.0 * beta) - 1.0
end

# B. Spec 1: Observed value as proxy (Naive OLS)
m1          = lm(@formula(g ~ 0 + g_sq), df_reg)
gamma_1     = extract_gamma(m1)

# C. Spec 2: Individual lagged variance
m2          = lm(@formula(g ~ 0 +g_sq_lag), df_reg)
gamma_2     = extract_gamma(m2)

# D. Spec 3: Average age variance (Contemporaneous)
m3          = lm(@formula(g ~ 0 +Age_Var_g), df_reg)
gamma_3     = extract_gamma(m3)

# E. Spec 4: Lagged average age variance (The correct instrument)
# First stage
fs          = lm(@formula(g_sq ~ 0 + Lag_Age_Var_g), df_reg)
df_reg.g_sq_hat = predict(fs)

# Second stage
m4          = lm(@formula(g ~ 0 + g_sq_hat), df_reg)
gamma_4     = (2.0 *  coef(m4)[1]) - 1.0

# F. Print the results 
println("True γ:              ", round(true_gamma, digits=3))
println("1. Proxy (Naive OLS):  ", round(gamma_1, digits=3))
println("2. Indiv Lagged Var:   ", round(gamma_2, digits=3))
println("3. Contemp Age Var:    ", round(gamma_3, digits=3))
println("4. Lagged Age Var (IV):", round(gamma_4, digits=3))

# 3. Plotting different simulated version 
# A. Calculate "Data" average consumption by age
df_data_means = combine(groupby(SyntheticPanel, :Age), :Obs_C => mean => :Mean_C)

# B. Initialise the plot 
plot_gamma = plot(df_data_means.Age, df_data_means.Mean_C, 
    linewidth = 3, color = :black, linestyle = :dot,
    label = "Noisy data", 
    xlabel = "Age", ylabel = "𝔼[c]",
    legend = :topleft, size = (800, 800), margin = 5Plots.mm)

# C. Rerun the model 
estimated_gammas = [gamma_1, gamma_2, gamma_3, gamma_4, true_gamma]
labels = ["1. Naive OLS", "2. Indiv. lagged", "3. Contemp. cohort", "4. Lagged cohort (Good IV)", "True γ"]
colors = [:red, :orange, :purple, :green, :navy]
styles = [:solid, :solid, :solid, :dash, :dash]
for i in 1:5

    # I. Computations 
    current_gamma = estimated_gammas[i]
    safe_gamma = max(current_gamma, 0.1) 
    test_params = ModelParameters(UsedParameters, γ = safe_gamma)
    test_endo   = fnSetUpEndo(test_params)
    fnNonStochasticSimulation!(test_params, test_endo; end_labour = true)
    mean_c_sim = zeros(test_params.T)
    for t in 1:test_params.T
        mean_c_sim[t] = sum(test_endo.Φ[t, :, :] .* test_endo.𝐂[t, :, :])
    end
    
    # II. Add to plot
    plot!(plot_gamma, 1:test_params.T, mean_c_sim, 
        linewidth = 2, color = colors[i], linestyle = styles[i],
        label = "$(labels[i]) (γ = $(round(safe_gamma, digits=2)))")
end

display(plot_gamma)
savefig(plot_gamma, "plots/gamma_estimation_comparison.pdf")