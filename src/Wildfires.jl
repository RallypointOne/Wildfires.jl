module Wildfires

import Random

include("Rothermel.jl")
include("LevelSet.jl")
include("SpreadModel.jl")
include("PINNTypes.jl")

using .PINNTypes: PINNConfig, PINNSolution
export PINNConfig, PINNSolution, train_pinn, predict_on_grid, predict_on_grid!

#-----------------------------------------------------------------------------# PINN API stubs
"""
    train_pinn(grid::LevelSet.LevelSetGrid, model, tspan; config=PINNConfig(), observations=nothing)

Train a Physics-Informed Neural Network to solve the fire spread level set PDE.

The PINN learns a function `phi_theta(x, y, t)` satisfying:

    dphi/dt + F(x,y,t)|nabla phi| = 0

where `F` is the spread rate from the `FireSpreadModel`.

Requires `Lux` to be loaded (triggers package extension).

# Arguments
- `grid` - `LevelSetGrid` providing domain geometry and initial condition
- `model` - Callable `model(t, x, y) -> spread_rate` (e.g. `FireSpreadModel`)
- `tspan` - Time interval `(t_start, t_end)`
- `config` - `PINNConfig` with training hyperparameters
- `observations` - Optional `(t, x, y, phi)` tuple of observation data

# Returns
A `PINNSolution` callable as `sol(t, x, y)`.

### Examples
```julia
sol = train_pinn(grid, model, (0.0, 50.0))
sol(25.0, 500.0, 500.0)  # evaluate at any point
```
"""
function train_pinn end

"""
    predict_on_grid(sol::PINNSolution, grid::LevelSet.LevelSetGrid, t)

Evaluate the trained PINN on every cell center of `grid` at time `t`.
Returns a matrix of phi values with the same dimensions as `grid`.

Requires `Lux` to be loaded (triggers package extension).
"""
function predict_on_grid end

"""
    predict_on_grid!(grid::LevelSet.LevelSetGrid, sol::PINNSolution, t)

In-place version of `predict_on_grid`: updates `grid.phi` and `grid.t`.

Requires `Lux` to be loaded (triggers package extension).
"""
function predict_on_grid! end

end # module Wildfires
