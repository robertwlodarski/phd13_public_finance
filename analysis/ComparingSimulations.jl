# Content
# 1. Wealth profiles 
# 2. Consumption profiles 
# 3. Labour supply 

# 1A. Function to extract the exact moments from both solvers
function fnComputeWealthMoments(params, endo)
    @unpack T, a⃗ = params

    # A. Monte Carlo moments)
    mean_MC = vec(mean(endo.Â, dims=2))
    med_MC  = vec(median(endo.Â, dims=2))

    # B. Histogram moments (Requires CDF math)
    mean_Hist = zeros(T)
    med_Hist  = zeros(T)

    for t in 1:T
        # 1. Mean: The dot product of the marginal distribution and the asset grid
        mean_Hist[t] = dot(endo.𝔼ʸΦ[t, :], a⃗)

        # 2. Median: Build the CDF and find where it crosses 0.5
        cdf = cumsum(endo.𝔼ʸΦ[t, :])
        idx = findfirst(x -> x >= 0.5, cdf)

        if idx == 1 || idx === nothing
            med_Hist[t] = a⃗[1]
        else
            # Linear interpolation between the two grid points for a perfectly smooth median
            dist_prev = 0.5 - cdf[idx-1]
            dist_next = cdf[idx] - 0.5
            weight    = dist_prev / (dist_prev + dist_next)
            med_Hist[t] = a⃗[idx-1] + weight * (a⃗[idx] - a⃗[idx-1])
        end
    end

    return mean_MC, med_MC, mean_Hist, med_Hist
end

# B. Extract the data using your solved inelastic model
mean_MC, med_MC, mean_Hist, med_Hist = fnComputeWealthMoments(UsedParameters, EndoMain)

# C. Extract the data using your solved inelastic model
mean_MC, med_MC, mean_Hist, med_Hist = fnComputeWealthMoments(UsedParameters, EndoMain)

# 1B. Plotting the wealth profiles 
# Generate the combined plot
plot_wealth = plot(1:UsedParameters.T, mean_Hist, 
    linewidth = 2, linestyle = :solid, color = :navy, 
    label = "𝔼[a] (Young)", 
    xlabel = "Age", ylabel = "a", 
    legend = :topleft, size = (800, 800), margin = 5Plots.mm)

# Add Mean MC
plot!(plot_wealth, 1:UsedParameters.T, mean_MC, 
    linewidth = 2, linestyle = :solid, color = :maroon, 
    label = "𝔼[a] (MC)")

# Add Median Histogram
plot!(plot_wealth, 1:UsedParameters.T, med_Hist, 
    linewidth = 2, linestyle = :dash, color = :navy, 
    label = "Q₅₀(a) (Young)")

# Add Median MC
plot!(plot_wealth, 1:UsedParameters.T, med_MC, 
    linewidth = 2, linestyle = :dash, color = :maroon, 
    label = "Q₅₀(a) (MC)")

# Display the final single-pane plot
display(plot_wealth)
savefig(plot_wealth, "plots/wealth_profile.pdf")

# 2A. Consumption: Extracting 
function fnComputeLogConsumptionMoments(params, endo)
    @unpack T = params

    # A. Monte Carlo moments
    log_C_MC  = log.(endo.Ĉ)
    mean_c_MC = vec(mean(log_C_MC, dims=2))
    med_c_MC  = vec(median(log_C_MC, dims=2))

    # B. Histogram moments
    mean_c_Hist = zeros(T)
    med_c_Hist  = zeros(T)

    for t in 1:T
        # I. Flatten the 2D mass and consumption policy matrices for period t
        mass_t     = vec(endo.Φ[t, :, :])
        cons_t     = vec(endo.𝐂[t, :, :])
        log_cons_t = log.(cons_t)

        # IIA. Mean: The dot product of the flattened mass and log consumption
        mean_c_Hist[t] = dot(mass_t, log_cons_t)

        # IIB. Median: Filter out zero masses to speed up the sort
        valid_idx      = mass_t .> 0.0
        mass_valid     = mass_t[valid_idx]
        log_cons_valid = log_cons_t[valid_idx]

        # IIC. Sort indices based on log consumption levels
        sort_idx     = sortperm(log_cons_valid)
        sorted_log_c = log_cons_valid[sort_idx]
        sorted_mass  = mass_valid[sort_idx]

        # IID. Build CDF and find the median threshold
        cdf = cumsum(sorted_mass)
        idx = findfirst(x -> x >= 0.5, cdf)
        
        if idx == 1 || idx === nothing
            med_c_Hist[t] = sorted_log_c[1]
        else
            # IIE. Linear interpolation for a perfectly smooth median
            dist_prev = 0.5 - cdf[idx-1]
            dist_next = cdf[idx] - 0.5
            weight    = dist_prev / (dist_prev + dist_next)
            med_c_Hist[t] = sorted_log_c[idx-1] + weight * (sorted_log_c[idx] - sorted_log_c[idx-1])
        end
    end

    return mean_c_MC, med_c_MC, mean_c_Hist, med_c_Hist
end

# 2B. Plot the consumption profile 
# A. Calculate the moments 
mean_c_MC, med_c_MC, mean_c_Hist, med_c_Hist = fnComputeLogConsumptionMoments(UsedParameters, EndoMain)

# B. Initialise the plot with your exact aesthetic preferences
plot_cons = plot(1:UsedParameters.T, mean_c_Hist, 
    linewidth = 2, linestyle = :solid, color = :navy, 
    label = "𝔼[log(c)] (Young)", 
    xlabel = "Age", ylabel = "log(c)", 
    legend = :topleft, size = (800, 800), margin = 5Plots.mm)

# C. Add mean MC
plot!(plot_cons, 1:UsedParameters.T, mean_c_MC, 
    linewidth = 2, linestyle = :solid, color = :maroon, 
    label = "𝔼[log(c)] (MC)")

# D. Add median histogram
plot!(plot_cons, 1:UsedParameters.T, med_c_Hist, 
    linewidth = 2, linestyle = :dash, color = :navy, 
    label = "Q₅₀(log(c)) (Young)")

# E. Add median MC
plot!(plot_cons, 1:UsedParameters.T, med_c_MC, 
    linewidth = 2, linestyle = :dash, color = :maroon, 
    label = "Q₅₀(log(c)) (MC)")

# F. Display and save
display(plot_cons)
savefig(plot_cons, "plots/log_consumption_profile.pdf")

# 3A. Labour supply: Compute the averages 
function fnComputeLabourMoments(params, endo)
    @unpack T = params

    # A. Monte Carlo moments
    mean_n_MC = vec(mean(endo.N̂, dims=2))

    # B. Histogram Moments
    mean_n_Hist = zeros(T)
    for t in 1:T
        mean_n_Hist[t] = sum(endo.Φ[t, :, :] .* endo.𝐍[t, :, :])
    end
    return mean_n_MC, mean_n_Hist
end

#3B. Plot 
# A. Calculate the moments using the elastic labor model!
mean_n_MC, mean_n_Hist = fnComputeLabourMoments(UsedParameters, EndoMain)

# B. Initialise the plot
plot_labour = plot(1:UsedParameters.T, mean_n_Hist, 
    linewidth = 2, linestyle = :solid, color = :navy, 
    label = "𝔼[n] (Young)", 
    xlabel = "Age", ylabel = "n", 
    legend = :bottomleft, size = (800, 800), margin = 5Plots.mm,
    ylims = (0.0, 1.05)) # Lock the y-axis to standard probability bounds

# C. Add Mean MC
plot!(plot_labour, 1:UsedParameters.T, mean_n_MC, 
    linewidth = 2, linestyle = :dash, color = :maroon, 
    label = "𝔼[n] (MC)")

# Display and save (using the relative path to step out of 'analysis' and into 'plots')
display(plot_labour)
savefig(plot_labour, "plots/labour_participation.pdf")

# 4. Inequality measures: Computations
# A. Helper functions for histogram math
function fnGiniHist(x_grid, p_mass)
    μ = sum(x_grid .* p_mass)
    if μ <= 0.0
        return 0.0
    end
    
    gini_sum = 0.0
    for i in eachindex(x_grid)
        for j in eachindex(x_grid)
            gini_sum += p_mass[i] * p_mass[j] * abs(x_grid[i] - x_grid[j])
        end
    end
    return gini_sum / (2.0 * μ)
end
function fnQuantileHist(x_grid, p_mass, target)
    cdf = cumsum(p_mass)
    idx = findfirst(c -> c >= target, cdf)
    
    if idx == 1 || idx === nothing
        return x_grid[1]
    end
    
    # Linear interpolation within the probability bin
    weight = (target - cdf[idx-1]) / p_mass[idx]
    return x_grid[idx-1] + weight * (x_grid[idx] - x_grid[idx-1])
end

# B. Main function 
function fnComputeInequality(params, endo)
    
    # I. Unpacking business
    @unpack T, a⃗, y⃗, ν⃗ = params

    # II. Preallocate vectors for the 6 lines
    gini_A   = zeros(T)
    gini_Y   = zeros(T)
    p10_90_A = zeros(T)
    p10_50_A = zeros(T)
    p10_90_Y = zeros(T)
    p10_50_Y = zeros(T)

    # III. Pre-calculate stationary income moments 
    # Since income is exogenous, its distribution doesn't change over time!
    gini_Y_stat   = fnGiniHist(y⃗, ν⃗)
    p10_Y_stat    = fnQuantileHist(y⃗, ν⃗, 0.10)
    p50_Y_stat    = fnQuantileHist(y⃗, ν⃗, 0.50)
    p90_Y_stat    = fnQuantileHist(y⃗, ν⃗, 0.90)
    
    p10_90_Y_stat = p90_Y_stat > 0.0 ? p10_Y_stat / p90_Y_stat : 0.0
    p10_50_Y_stat = p50_Y_stat > 0.0 ? p10_Y_stat / p50_Y_stat : 0.0

    # IV. Start the loop
    for t in 1:T
        # IVA. Wealth Moments 
        mass_A      = endo.𝔼ʸΦ[t, :]
        gini_A[t]   = fnGiniHist(a⃗, mass_A)
        
        # IVB. Calculate percentiles 
        p10_A       = fnQuantileHist(a⃗, mass_A, 0.10)
        p50_A       = fnQuantileHist(a⃗, mass_A, 0.50)
        p90_A       = fnQuantileHist(a⃗, mass_A, 0.90)
        
        # IVC. Protect against division by zero 
        p10_90_A[t] = p90_A > 0.0 ? p10_A / p90_A : 0.0
        p10_50_A[t] = p50_A > 0.0 ? p10_A / p50_A : 0.0

        # IVD. Income moments (Assigning the constant stationary values)
        gini_Y[t]   = gini_Y_stat
        p10_90_Y[t] = p10_90_Y_stat
        p10_50_Y[t] = p10_50_Y_stat
    end
    
    return gini_A, gini_Y, p10_90_A, p10_50_A, p10_90_Y, p10_50_Y
end

# 4B. Plot inequality measures 
# A. Compute
gini_A, gini_Y, p10_90_A, p10_50_A, p10_90_Y, p10_50_Y = fnComputeInequality(UsedParameters, EndoMain)

# B. Initialise the plot with the Gini solid lines
plot_ineq = plot(1:UsedParameters.T, gini_A, 
    linewidth = 2, linestyle = :solid, color = :maroon, 
    label = "Gini (a)", 
    xlabel = "Age", ylabel = "Ratio ∈ [0, 1]", 
    ylims = (0.0, 0.7),
    xlims = (2,40),
    legend = :outertopright, 
    size = (800, 800), margin = 5Plots.mm)
annotate!(plot_ineq, 3, 0.25, text("t=1 skipped", :left, 10, :black))

plot!(plot_ineq, 1:UsedParameters.T, gini_Y, 
    linewidth = 2, linestyle = :solid, color = :navy, 
    label = "Gini (y)")

# Add the P10 / P90 ratios (Dashed)
plot!(plot_ineq, 1:UsedParameters.T, p10_90_A, 
    linewidth = 2, linestyle = :dash, color = :maroon, 
    label = "P₁₀/P₉₀ (a)")

plot!(plot_ineq, 1:UsedParameters.T, p10_90_Y, 
    linewidth = 2, linestyle = :dash, color = :navy, 
    label = "P₁₀/P₉₀ (y)")

# Add the P10 / P50 ratios (Dotted)
plot!(plot_ineq, 1:UsedParameters.T, p10_50_A, 
    linewidth = 2, linestyle = :dot, color = :maroon, 
    label = "P₁₀/P₅₀ (a)")

plot!(plot_ineq, 1:UsedParameters.T, p10_50_Y, 
    linewidth = 2, linestyle = :dot, color = :navy, 
    label = "P₁₀/P₅₀ (y)")

# Display and save
display(plot_ineq)
savefig(plot_ineq, "plots/inequality_dynamics.pdf")