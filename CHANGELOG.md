# Changelog

## v0.1.0

Initial release of Wildfires.jl.

### Features

- **Rothermel model**: Full implementation of the Rothermel (1972) surface fire spread model with all 13 NFFL fuel models from Anderson (1982)
- **Level set method**: 2D fire front propagation using a level set approach with Godunov upwind scheme and reinitialization
- **Composable spread model**: `FireSpreadModel` with pluggable components (`UniformWind`, `UniformMoisture`, `DynamicMoisture`, `FlatTerrain`, `GriddedTerrain`)
- **GPU acceleration**: Backend-agnostic GPU support (CUDA, Metal, ROCm) via KernelAbstractions.jl package extension for `advance!`, `reinitialize!`, and moisture update kernels
- **PINN solver**: Physics-informed neural network extension for level set equation solving with hard IC constraints (via Lux.jl)
- **Makie integration**: `fireplot` recipe and `convert_arguments` for direct plotting of `LevelSetGrid`
- **Quarto documentation**: Full documentation site with getting started guide, API reference, and GPU/PINN pages
