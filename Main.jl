# PHD13 Public finance
# Replication: Life cycle model 
# March 2026

## 1. Packages & load functions 
using Parameters, QuantEcon, LinearAlgebra, Roots, Printf, Plots, Distributions 
include("scripts/ModelInfrastructure.jl")
include("scripts/Functions.jl")

## 2. Solve for steady state 
# p̂   = @time fnSolveSteadyState!(UsedParameters,Endo)
