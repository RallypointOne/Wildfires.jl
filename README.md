[![CI](https://github.com/RallypointOne/Wildfires.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/RallypointOne/Wildfires.jl/actions/workflows/CI.yml)
[![Docs Build](https://github.com/RallypointOne/Wildfires.jl/actions/workflows/Docs.yml/badge.svg)](https://github.com/RallypointOne/Wildfires.jl/actions/workflows/Docs.yml)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue)](https://RallypointOne.github.io/Wildfires.jl/stable/)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue)](https://RallypointOne.github.io/Wildfires.jl/dev/)

# Wildfires.jl

A Julia package for wildfire modeling and simulation, built around three core modules:

- **Rothermel** — The Rothermel (1972) surface fire spread model with the 13 standard NFFL fuel models from Anderson (1982).
- **Level Set** — A level set method for simulating 2D fire front propagation driven by spatially varying spread rates.
- **Spread Model** — Composable, differentiable components (wind, moisture, terrain) that drive level set simulations via `FireSpreadModel`.

Additional capabilities are available through package extensions:

- **GPU Acceleration** — Backend-agnostic GPU support (CUDA, Metal, ROCm) via KernelAbstractions.jl.
- **PINN Solver** — Physics-informed neural network for solving the level set PDE (via Lux.jl).

## Installation

```julia
using Pkg
Pkg.add("Wildfires")
```

## Quick Example

```julia
using Wildfires
using Wildfires.Rothermel
using Wildfires.LevelSet
using Wildfires.SpreadModel

# Build a fire spread model from components
moisture = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(moisture), FlatTerrain())

# Create a grid, ignite, and simulate
grid = LevelSetGrid(200, 200, dx=30.0)
ignite!(grid, 3000.0, 3000.0, 200.0)
simulate!(grid, model, steps=500, dt=0.5)

# Visualize (requires Makie)
# fireplot(grid)
```
