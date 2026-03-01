module Wildfires

include("Rothermel.jl")
include("LevelSet.jl")
include("SpreadModel.jl")
include("PINNTypes.jl")

using .PINNTypes: AbstractPINNConfig, PINNConfig, NeuralPDEConfig, PINNSolution
export AbstractPINNConfig, PINNConfig, NeuralPDEConfig, PINNSolution
export train_pinn, predict_on_grid, predict_on_grid!
export fireplot, fireplot!, firegif

#-----------------------------------------------------------------------------# Makie stubs (implemented in WildfiresMakieExt)
"""
    fireplot(grid::LevelSetGrid; residence_time=nothing, frontcolor=:black, frontlinewidth=2.0)

Plot a `LevelSetGrid` as a heatmap with the fire front (`φ = 0`) overlaid as a contour line.
Returns a `Makie.Figure`.

When `residence_time` is provided, uses a burnout-aware colormap where burned cells
transition from yellow (just ignited) → red → black (burnt out) and unburned cells
transition from white (near front) → green (far away).

Without `residence_time`, falls back to a symmetric `φ` heatmap with `:RdYlGn` colormap.

Requires `Makie` (or a backend like `CairoMakie` / `GLMakie`) to be loaded.

### Examples
```julia
using CairoMakie
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)
fireplot(grid)
fireplot(grid; residence_time=0.005)
```
"""
function fireplot end

"""
    fireplot!(ax, grid::LevelSetGrid; residence_time=nothing, frontcolor=:black, frontlinewidth=2.0)

In-place version of `fireplot`: draws into an existing `Axis`.

Requires `Makie` (or a backend like `CairoMakie` / `GLMakie`) to be loaded.
"""
function fireplot! end

"""
    firegif(path, trace::Trace, grid::LevelSetGrid; residence_time=nothing, framerate=15, frontcolor=:black, frontlinewidth=2.0)

Create an animated GIF of fire spread from a [`Trace`](@ref) recorded during [`simulate!`](@ref).

Uses the same visualization style as [`fireplot!`](@ref).

Requires `Makie` (or a backend like `CairoMakie` / `GLMakie`) to be loaded.

### Examples
```julia
using CairoMakie

grid = LevelSetGrid(200, 200, dx=30.0)
ignite!(grid, 3000.0, 3000.0, 50.0)
trace = Trace(grid, 5)
simulate!(grid, model, steps=100, trace=trace)

firegif("fire.gif", trace, grid)
firegif("fire.gif", trace, grid; residence_time=0.005)
```
"""
function firegif end

#-----------------------------------------------------------------------------# PINN API stubs
"""
    train_pinn(grid, model, tspan; config=PINNConfig(), ...)
    train_pinn(grid, model, tspan, config; ...)

Train a Physics-Informed Neural Network to solve the fire spread level set PDE.

The PINN learns a function `phi_theta(x, y, t)` satisfying:

    dphi/dt + F(x,y,t)|nabla phi| = 0

where `F` is the spread rate from the `FireSpreadModel`.

The solver backend is selected by the `config` type:
- `PINNConfig` — custom Lux solver with hard IC constraint (requires `Lux`)
- `NeuralPDEConfig` — NeuralPDE.jl symbolic solver (requires `NeuralPDE`)

# Arguments
- `grid` - `LevelSetGrid` providing domain geometry and initial condition
- `model` - Callable `model(t, x, y) -> spread_rate` (e.g. `FireSpreadModel`)
- `tspan` - Time interval `(t_start, t_end)`
- `config` - `PINNConfig` or `NeuralPDEConfig` with training hyperparameters
- `observations` - Optional `(t, x, y, phi)` tuple of observation data (Lux backend only)
- `lbfgs_optimizer` - Optimizer for L-BFGS refinement phase, e.g. `OptimizationOptimJL.LBFGS()` (Lux backend only, requires `lbfgs_epochs > 0` in config)

# Returns
A `PINNSolution` callable as `sol(t, x, y)`.

### Examples
```julia
# Custom Lux backend (default)
sol = train_pinn(grid, model, (0.0, 50.0))

# NeuralPDE backend
sol = train_pinn(grid, model, (0.0, 50.0); config=NeuralPDEConfig())
```
"""
function train_pinn(grid::LevelSet.LevelSetGrid, model, tspan::Tuple;
                    config::AbstractPINNConfig=PINNConfig(), kwargs...)
    train_pinn(grid, model, tspan, config; kwargs...)
end

function train_pinn(grid::LevelSet.LevelSetGrid, model, tspan::Tuple,
                    config::AbstractPINNConfig; kwargs...)
    error("No PINN backend loaded. Load the Lux or NeuralPDE extension.")
end

"""
    predict_on_grid(sol::PINNSolution, grid::LevelSet.LevelSetGrid, t)

Evaluate the trained PINN on every cell center of `grid` at time `t`.
Returns a matrix of phi values with the same dimensions as `grid`.
"""
function predict_on_grid end

"""
    predict_on_grid!(grid::LevelSet.LevelSetGrid, sol::PINNSolution, t)

In-place version of `predict_on_grid`: updates `grid.phi` and `grid.t`.
"""
function predict_on_grid! end

#-----------------------------------------------------------------------------# PINNSolution fallback callable + predict
# Base-module fallbacks: work when model stores a callable (e.g. NeuralPDE backend).
# The Lux extension overrides these with its own methods.

function (sol::PINNSolution)(t, x, y)
    if applicable(sol.model, t, x, y)
        return sol.model(t, x, y)
    end
    error("PINNSolution is not callable. Load the appropriate PINN extension (Lux or NeuralPDE).")
end

function predict_on_grid(sol::PINNSolution, grid::LevelSet.LevelSetGrid, t)
    xs = LevelSet.xcoords(grid)
    ys = LevelSet.ycoords(grid)
    φ = Matrix{Float64}(undef, length(ys), length(xs))
    for j in eachindex(xs), i in eachindex(ys)
        φ[i, j] = sol(t, xs[j], ys[i])
    end
    φ
end

function predict_on_grid!(grid::LevelSet.LevelSetGrid, sol::PINNSolution, t)
    xs = LevelSet.xcoords(grid)
    ys = LevelSet.ycoords(grid)
    for j in eachindex(xs), i in eachindex(ys)
        grid.φ[i, j] = sol(t, xs[j], ys[i])
    end
    grid.t = t
    grid
end

end # module Wildfires
