# PHD13 Public Finance in Julia

This repository translates the original Matlab code from the PHD13 Public Finance course into Julia. Taking advantage of Julia's performance and multiple dispatch, this codebase also extends the original dynamic programming framework to include a **binary labor supply decision**.

## Font Installation (Highly Recommended)

This project uses Unicode characters and mathematical symbols to make the Julia code as readable and closely aligned to the formal notation as possible. 

To ensure these special characters render correctly in your IDE or terminal, please install the **JuliaMono** typeface:

* **Download and Instructions:** [JuliaMono GitHub Repository](https://github.com/cormullion/juliamono)
* **Setup:** Once downloaded, install the font files to your system and update your code editor's font family settings to `JuliaMono`.

## Getting Started

To help you navigate the codebase, here is a quick breakdown of the project structure:

*   **`Main.jl`:** The central entry point for the project. Everything, from environment setup to running the full simulation, can be executed directly from this file.
*   **`scripts/`:** The core engine room. This folder houses all the underlying source code, including `ModelInfrastructure.jl` (for grids and parameter structs) and `FunctionsModel.jl` (for the Bellman and Euler equations).
*   **`analysis/`:** The analytical hub. Contains all the scripts used for post-simulation processing, such as generating plots and running regressions on the simulated data.
*   **`note/`:** The theoretical companion. This directory contains summaries of the course content and the formal answers to the assignment questions. 

To run the project, ensure your Julia environment is instantiated, and then simply execute `Main.jl`.
