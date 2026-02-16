module PINNTypes

export PINNConfig, PINNSolution

#-----------------------------------------------------------------------------# PINNConfig
"""
    PINNConfig(; kwargs...)

Training hyperparameters for the Physics-Informed Neural Network solver.

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
- `max_epochs::Int` - Maximum training epochs (default `5000`)
- `resample_every::Int` - Resample collocation points every N epochs (default `500`)

### Examples
```julia
config = PINNConfig(hidden_dims=[128, 128], max_epochs=10000)
```
"""
Base.@kwdef struct PINNConfig
    hidden_dims::Vector{Int} = [64, 64, 64]
    activation::Symbol = :tanh
    n_interior::Int = 5000
    n_boundary::Int = 500
    lambda_pde::Float64 = 1.0
    lambda_bc::Float64 = 1.0
    lambda_data::Float64 = 1.0
    learning_rate::Float64 = 1e-3
    max_epochs::Int = 5000
    resample_every::Int = 500
end

#-----------------------------------------------------------------------------# PINNSolution
"""
    PINNSolution

Trained PINN model. Callable as `sol(t, x, y)` to evaluate the level set function.

# Fields
- `model` - Lux neural network chain
- `parameters` - Trained parameters (ComponentArray)
- `state` - Lux model state
- `config::PINNConfig` - Training configuration
- `loss_history::Vector{Float64}` - Loss at each epoch
- `domain::NamedTuple` - `(tspan, xspan, yspan, phi_scale)` for input normalization
- `grid_ic` - Initial condition grid (for hard IC constraint decomposition)

### Examples
```julia
phi = sol(10.0, 500.0, 500.0)  # evaluate at t=10, x=500, y=500
```
"""
mutable struct PINNSolution
    model::Any
    parameters::Any
    state::Any
    config::PINNConfig
    loss_history::Vector{Float64}
    domain::NamedTuple
    grid_ic::Any
end

function Base.show(io::IO, sol::PINNSolution)
    n = length(sol.loss_history)
    final_loss = n > 0 ? round(sol.loss_history[end], sigdigits=4) : NaN
    print(io, "PINNSolution(epochs=$n, final_loss=$final_loss)")
end

end # module
