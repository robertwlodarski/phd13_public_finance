# PHD13 Public finance in Julia

This repository translates the original Matlab code from the PHD13 Public Finance course into Julia. Taking advantage of Julia's performance and multiple dispatch, this codebase also extends the original dynamic programming framework to include a **binary labor supply decision**.

## Font installation

This project uses Unicode characters and mathematical symbols to make the Julia code as readable and closely aligned to the formal notation as possible. 

To ensure these special characters render correctly in your IDE or terminal, please install the **JuliaMono** typeface:

* **Download and instructions:** [JuliaMono GitHub Repository](https://github.com/cormullion/juliamono)
* **Setup:** Once downloaded, install the font files to your system and update your code editor's font family settings to `JuliaMono`.

## Notation 

I rely on the Unicode characters to make the Julia code as closely aligned to the formal notation as possible. 
In so doing, I follow the convention below in defining the modelled objects.

1. Roman alphabet: Generic objects
2. Greek letters: Parameters
3. Symbols with arrows: Vectors
4. Bold symbols: Value and policy functions
5. Typeset symbols: Measures


## Getting started

To help you navigate the codebase, here is a quick breakdown of the project structure:

*   **`Main.jl`:** The central entry point for the project. Everything, from environment setup to running the full simulation, can be executed directly from this file.
*   **`scripts/`:** The core engine room. This folder houses all the underlying source code, including `ModelInfrastructure.jl` (for grids and parameter structs) and `FunctionsModel.jl` (for the Bellman and Euler equations).
*   **`analysis/`:** The analytical hub. Contains all the scripts used for post-simulation processing, such as generating plots and running regressions on the simulated data.
*   **`note/`:** The theoretical companion. This directory contains summaries of the course content and the formal answers to the assignment questions. 

To run the project, ensure your Julia environment is instantiated, and then simply execute `Main.jl`.

## Improvements on the original code

* The new code relies heavily on Young (2010)'s non-stochastic simulation rather than the default Monte Carlo (MC) simulation, which is prone to noise. 
* The code still produces a MC output that can be used for any regressions. 
* The script nests the models without and with endogenous (extensive margin) labour supply. 