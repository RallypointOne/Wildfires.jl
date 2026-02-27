module PINNTypes

export AbstractPINNConfig, PINNConfig, NeuralPDEConfig, PINNSolution

#-----------------------------------------------------------------------------# AbstractPINNConfig
"""
    AbstractPINNConfig

Supertype for PINN training configurations.  Subtypes select the solver backend:

- `PINNConfig` — custom Lux solver with hard IC constraint
- `NeuralPDEConfig` — NeuralPDE.jl / ModelingToolkit solver
"""
abstract type AbstractPINNConfig end

#-----------------------------------------------------------------------------# PINNConfig
"""
    PINNConfig(; kwargs...)

Training hyperparameters for the custom Lux-based PINN solver.

The initial condition is enforced exactly via a hard constraint decomposition
(no IC loss term or IC collocation points needed).

# Keyword Arguments
- `hidden_dims::Vector{Int}` - Hidden layer sizes (default `[64, 64, 64]`)
- `activation::Symbol` - Activation function (default `:tanh`)
- `n_interior::Int` - PDE collocation points (default `5000`)
- `n_boundary::Int` - Boundary condition points (default `500`)
- `lambda_pde::Float64` - PDE loss weight (default `1.0`)
- `lambda_bc::Float64` - BC loss weight (default `1.0`)
- `lambda_data::Float64` - Data loss weight (default `1.0`)
- `learning_rate::Float64` - Adam learning rate (default `1e-3`)
- `max_epochs::Int` - Maximum training epochs (default `10000`)
- `resample_every::Int` - Resample collocation points every N epochs (default `500`)
- `lbfgs_epochs::Int` - L-BFGS refinement epochs after Adam (default `0`, disabled)
- `importance_sampling::Bool` - Concentrate points near fire front (default `false`)
- `float32::Bool` - Use Float32 for NN weights (default `false`)

### Examples
```julia
config = PINNConfig(hidden_dims=[128, 128], max_epochs=10000)
```
"""
Base.@kwdef struct PINNConfig <: AbstractPINNConfig
    hidden_dims::Vector{Int} = [64, 64, 64]
    activation::Symbol = :tanh
    n_interior::Int = 5000
    n_boundary::Int = 500
    lambda_pde::Float64 = 1.0
    lambda_bc::Float64 = 1.0
    lambda_data::Float64 = 1.0
    learning_rate::Float64 = 1e-3
    max_epochs::Int = 10000
    resample_every::Int = 500
    lbfgs_epochs::Int = 0
    importance_sampling::Bool = false
    float32::Bool = false
end

#-----------------------------------------------------------------------------# NeuralPDEConfig
"""
    NeuralPDEConfig(; kwargs...)

Training hyperparameters for the NeuralPDE.jl PINN solver.

Uses ModelingToolkit symbolic PDE definition with `PhysicsInformedNN` discretization.
IC and BC are enforced as soft constraints (loss terms).

Requires `NeuralPDE` and `ModelingToolkit` to be loaded (triggers package extension).

# Keyword Arguments
- `hidden_dims::Vector{Int}` - Hidden layer sizes (default `[16, 16]`)
- `activation::Symbol` - Activation function (default `:σ`)
- `strategy::Symbol` - Training strategy: `:grid`, `:stochastic` (default `:grid`)
- `grid_step::Float64` - Grid spacing for `GridTraining` (default `0.1`)
- `max_epochs::Int` - Maximum training iterations (default `1000`)
- `optimizer::Symbol` - Optimizer: `:lbfgs`, `:bfgs` (default `:lbfgs`)
- `learning_rate::Float64` - Learning rate for Adam (default `1e-2`)

### Examples
```julia
config = NeuralPDEConfig(hidden_dims=[32, 32], max_epochs=2000)
sol = train_pinn(grid, model, tspan; config=config)
```
"""
Base.@kwdef struct NeuralPDEConfig <: AbstractPINNConfig
    hidden_dims::Vector{Int} = [16, 16]
    activation::Symbol = :σ
    strategy::Symbol = :grid
    grid_step::Float64 = 0.1
    max_epochs::Int = 1000
    optimizer::Symbol = :lbfgs
    learning_rate::Float64 = 1e-2
end

#-----------------------------------------------------------------------------# PINNSolution
"""
    PINNSolution

Trained PINN model. Callable as `sol(t, x, y)` to evaluate the level set function.

# Fields
- `model` - Neural network or callable evaluator
- `parameters` - Trained parameters
- `state` - Model state (backend-specific)
- `config::AbstractPINNConfig` - Training configuration
- `loss_history::Vector{Float64}` - Loss at each epoch
- `domain::NamedTuple` - `(tspan, xspan, yspan, phi_scale)` for input normalization
- `grid_ic` - Initial condition grid

### Examples
```julia
phi = sol(10.0, 500.0, 500.0)  # evaluate at t=10, x=500, y=500
```
"""
mutable struct PINNSolution
    model::Any
    parameters::Any
    state::Any
    config::AbstractPINNConfig
    loss_history::Vector{Float64}
    domain::NamedTuple
    grid_ic::Any
end

function Base.show(io::IO, sol::PINNSolution)
    n = length(sol.loss_history)
    final_loss = n > 0 ? round(sol.loss_history[end], sigdigits=4) : NaN
    backend = sol.config isa PINNConfig ? "Lux" : "NeuralPDE"
    print(io, "PINNSolution{$backend}(epochs=$n, final_loss=$final_loss)")
end

end # module
