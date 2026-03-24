# PHD13 Public finance
# Replication: Life cycle model 
# March 2026

# PHD13 Public finance
# Replication: Life cycle model 
# March 2026

## 1. Packages & load functions 
using Parameters, QuantEcon, LinearAlgebra, Roots, Printf, Plots, Distributions, StatsBase, Random, Dierckx 
include("scripts/ModelInfrastructure.jl")
include("scripts/Functions.jl")

## 2. Solve two versions of the model
@time fnNonStochasticSimulation!(UsedParameters,EndoInelasticLab;end_labour=false)
@time fnMonteCarlo!(UsedParameters, EndoInelasticLab; end_labour = false)
@time fnNonStochasticSimulation!(UsedParameters,EndoMain;end_labour=true)
@time fnMonteCarlo!(UsedParameters, EndoMain; end_labour = true)
