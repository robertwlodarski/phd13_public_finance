# PHD20 Firms and sectors in the macroeconomy
# Replication: Hopenhayn and Rogerson (1993) 
# February 2026

## 1. Packages & load functions 
using Parameters, QuantEcon, LinearAlgebra, Roots, Printf, Plots, Distributions 
include("functions/ModelInfrastructure.jl")
include("functions/Functions.jl")

## 2. Solve for steady state 
p̂   = @time fnSolveSteadyState!(UsedParameters,Endo)
