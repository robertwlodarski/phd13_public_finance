# Content:
# 1. Objective function 
# 2. SMM wrapper 
# 3. Plot the fit
# 4. Labour policies 

# 1. Objective function 
function fnObjectiveChi_MaxDev(χ_guess, σ_val, target_moments, base_params, endo, error_history, chi_history)
    
    # A. Reconstruct parameters 
    safe_χ                  = max(χ_guess[1], 0.001)
    test_params             = ModelParameters(base_params, χ = safe_χ, σ = σ_val)
    
    # B. Solve the model
    fnNonStochasticSimulation!(test_params, endo; end_labour = true)
    
    # C. Compute expected employment by age
    model_moments = zeros(test_params.T)
    for t in 1:test_params.T
        model_moments[t]    = sum(endo.Φ[t, :, :] .* endo.𝐍[t, :, :])
    end
    
    # D. Calculate the error 
    max_dev                 = maximum(abs.(model_moments .- target_moments))
    
    # E. Record the error and chi for the convergence plot
    push!(error_history, max_dev)
    push!(chi_history, safe_χ)
    return max_dev
end

# 2. SMM wrapper 
function fnCalibrateChi(df_data, base_params, endo; σ_val = 4.0, initial_chi = 10.0)
    
    # A. Extract Real Data Moments
    df_grouped      = combine(groupby(df_data, :Age), :Employed => mean => :EmpRate)
    target_moments  = df_grouped.EmpRate
    
    # B. Initialise the tracking array
    error_history   = Float64[]
    chi_history     = Float64[]
    # C. Define the closure
    f_obj           = x -> fnObjectiveChi_MaxDev(x, σ_val, target_moments, base_params, endo, error_history, chi_history)
    
    # D. Run 
    println("Starting SMM... Initial χ = ", initial_chi, " | Fixed σ = ", σ_val)
    @time res = optimize(f_obj, [initial_chi], NelderMead())
    est_chi = max(Optim.minimizer(res)[1], 0.001)
    println("Calibration Complete. Estimated χ: ", round(est_chi, digits=4))
    
    # E. Plot 1: Convergence History
    plot_chi = [c > 11.0 ? NaN : c for c in chi_history]
    p_conv = plot(1:length(error_history), error_history,
        linewidth = 2, color = :navy, 
        xlabel = "Iteration", 
        ylabel = "max(|ϵ|)",
        legend = :none, size = (800, 800), margin = 5Plots.mm)
    p_twin = twinx(p_conv)
    plot!(p_twin, 1:length(plot_chi), plot_chi, 
        seriestype = :scatter, markersize = 4, color = :maroon, 
        ylabel = "Tested χ", ylims = (0.0, 11.0),
        legend = :none)
    display(p_conv)
    savefig(p_conv, "plots/smm_convergence_$σ_val.pdf")
    return est_chi, target_moments
end

# Execute
chosen_sigma = 0.5 
estimated_chi, empirical_employment = fnCalibrateChi(SyntheticPanel, UsedParameters, EndoMain; σ_val = chosen_sigma, initial_chi = 10.0)

# 3: Plot the fit
# A. Re-solve with the optimal estimated parameters
est_params = ModelParameters(UsedParameters, χ = estimated_chi, σ = chosen_sigma)
fnNonStochasticSimulation!(est_params, EndoMain; end_labour = true)

# B. Extract the model's optimized employment moments
est_moments = zeros(est_params.T)
for t in 1:est_params.T
    est_moments[t] = sum(EndoMain.Φ[t, :, :] .* EndoMain.𝐍[t, :, :])
end

# C. Plot 
p_fit = plot(1:est_params.T, empirical_employment, 
    linewidth = 3, color = :black, linestyle = :dot,
    label = "Synthetic data", 
    xlabel = "Age", ylabel = "𝔼[n]",
    legend = :bottomleft, size = (800, 800), margin = 5Plots.mm,
    ylims = (0.0, 1.05))

plot!(p_fit, 1:est_params.T, est_moments, 
    linewidth = 2, color = :maroon, 
    label = "Estimated (χ = $(round(estimated_chi, digits=2)))")
annotate!(p_fit, 3, 0.35, text("True χ = $(round(UsedParameters.χ, digits=2))", :left, 10, :black))
display(p_fit)

# 4. Labour supply policies 
policy_matrix_est   = Float64.(EndoMain.𝐍[20, :, :])

# A. Re-solve the model using the EXACT True Parameters to get the real DGP policy
true_params         = UsedParameters
fnNonStochasticSimulation!(true_params, EndoMain; end_labour = true)
policy_matrix_true  = Float64.(EndoMain.𝐍[20, :, :])

# B. Build the Left Pane (Estimated)
p_pol_left  = heatmap(est_params.a⃗, est_params.y⃗, policy_matrix_est, 
    color   = cgrad([:white, :navy]), 
    title   = "Estimated (χ = $(round(estimated_chi, digits=2)))",
    xlabel  = "Wealth (a)", ylabel = "Income (y)", 
    legend  = :none)

# C. Build the Right Pane (True Parameters)
p_pol_right = heatmap(true_params.a⃗, true_params.y⃗, policy_matrix_true, 
    color = cgrad([:white, :maroon]), 
    title = "True (χ = $(round(true_params.χ, digits=2)))",
    xlabel = "Wealth (a)", ylabel = "Income (y)", 
    legend = :none)

# D. Combine into a dual-pane layout
plot_policies = plot(p_pol_left, p_pol_right, layout = (1, 2), size = (800, 400), margin = 5Plots.mm)
display(plot_policies)
savefig(plot_policies, "plots/labour_policies.pdf")
