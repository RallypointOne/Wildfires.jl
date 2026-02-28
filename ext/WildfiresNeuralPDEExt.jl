module WildfiresNeuralPDEExt

using Wildfires
using Wildfires: NeuralPDEConfig, PINNSolution
using Wildfires.LevelSet: LevelSetGrid, xcoords, ycoords

using NeuralPDE
using ModelingToolkit
using DomainSets: ClosedInterval
using Lux
using Optimization
using OptimizationOptimJL

import Random

#-----------------------------------------------------------------------------# Bilinear interpolation for IC
function _bilinear_ic(φ_ic::Matrix{Float64}, x0::Float64, y0::Float64,
                      dx::Float64, dy::Float64, x::Float64, y::Float64)
    ny, nx = size(φ_ic)
    fx = clamp((x - x0) / dx, 0.0, Float64(nx - 1))
    fy = clamp((y - y0) / dy, 0.0, Float64(ny - 1))
    j0 = clamp(floor(Int, fx) + 1, 1, nx - 1)
    i0 = clamp(floor(Int, fy) + 1, 1, ny - 1)
    j1 = min(j0 + 1, nx)
    i1 = min(i0 + 1, ny)
    tx = fx - (j0 - 1)
    ty = fy - (i0 - 1)
    (1 - tx) * (1 - ty) * φ_ic[i0, j0] + tx * (1 - ty) * φ_ic[i0, j1] +
    (1 - tx) * ty * φ_ic[i1, j0] + tx * ty * φ_ic[i1, j1]
end

#-----------------------------------------------------------------------------# Module-level refs for @register_symbolic functions
const _SPREAD_REF = Ref{Any}((t, x, y) -> 0.0)
const _IC_REF = Ref{Any}((x, y) -> 0.0)

_spread_fn(t, x, y) = Float64(_SPREAD_REF[](t, x, y))
_ic_fn(x, y) = Float64(_IC_REF[](x, y))

@register_symbolic _spread_fn(t, x, y)
@register_symbolic _ic_fn(x, y)

#-----------------------------------------------------------------------------# Build Lux chain from config
function _build_chain(config::NeuralPDEConfig)
    act = _get_act(config.activation)
    dims = config.hidden_dims
    layers = Any[Dense(3 => dims[1], act)]
    for i in 2:length(dims)
        push!(layers, Dense(dims[i-1] => dims[i], act))
    end
    push!(layers, Dense(dims[end] => 1))
    Chain(layers...)
end

function _get_act(s::Symbol)
    s === :tanh && return tanh
    s === :σ && return sigmoid
    s === :sigmoid && return sigmoid
    s === :relu && return relu
    s === :gelu && return gelu
    error("Unknown activation: $s. Use :tanh, :σ, :sigmoid, :relu, or :gelu.")
end

#-----------------------------------------------------------------------------# train_pinn (NeuralPDE backend)
function Wildfires.train_pinn(grid::LevelSetGrid, spread_model, tspan::Tuple,
                              config::NeuralPDEConfig;
                              observations=nothing,
                              rng=Random.default_rng(),
                              verbose::Bool=true)
    xs_grid = xcoords(grid)
    ys_grid = ycoords(grid)

    phi_scale = maximum(abs, grid.φ)
    phi_scale == 0 && (phi_scale = one(eltype(grid.φ)))

    x_min = first(xs_grid) - step(xs_grid) / 2
    x_max = last(xs_grid) + step(xs_grid) / 2
    y_min = first(ys_grid) - step(ys_grid) / 2
    y_max = last(ys_grid) + step(ys_grid) / 2
    t_start, t_end = tspan

    domain_nt = (
        tspan = tspan,
        xspan = (x_min, x_max),
        yspan = (y_min, y_max),
        phi_scale = phi_scale,
    )

    # Snapshot IC
    ic_φ = copy(Matrix{Float64}(grid.φ))
    ic_x0 = Float64(first(xs_grid))
    ic_y0 = Float64(first(ys_grid))
    ic_dx = Float64(step(xs_grid))
    ic_dy = Float64(step(ys_grid))
    grid_ic = LevelSetGrid(copy(grid.φ), copy(grid.t_ignite), grid.dx, grid.dy, grid.x0, grid.y0, grid.t, grid.bc)

    # Set module-level refs for registered symbolic functions
    _SPREAD_REF[] = spread_model
    _IC_REF[] = (x, y) -> _bilinear_ic(ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, Float64(x), Float64(y))

    # Symbolic PDE definition
    @parameters t x y
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)

    ε = 1e-6
    eq = [Dt(u(t, x, y)) + _spread_fn(t, x, y) * sqrt(Dx(u(t, x, y))^2 + Dy(u(t, x, y))^2 + ε^2) ~ 0]
    bcs = [u(t_start, x, y) ~ _ic_fn(x, y)]
    domains = [
        t ∈ ClosedInterval(t_start, t_end),
        x ∈ ClosedInterval(x_min, x_max),
        y ∈ ClosedInterval(y_min, y_max),
    ]

    # Neural network
    chain = _build_chain(config)

    # Training strategy
    strategy = if config.strategy === :grid
        NeuralPDE.GridTraining(config.grid_step)
    elseif config.strategy === :stochastic
        NeuralPDE.StochasticTraining(100)
    else
        error("Unknown strategy: $(config.strategy). Use :grid or :stochastic.")
    end

    # Discretization and PDE system
    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bcs, domains, [t, x, y], [u(t, x, y)])
    prob = NeuralPDE.discretize(pde_system, discretization)

    # Optimizer
    opt = if config.optimizer === :lbfgs
        OptimizationOptimJL.LBFGS()
    elseif config.optimizer === :bfgs
        OptimizationOptimJL.BFGS()
    else
        error("Unknown optimizer: $(config.optimizer). NeuralPDE backend supports :lbfgs and :bfgs.")
    end

    # Training with loss tracking
    loss_history = Float64[]
    callback = function (state, l)
        push!(loss_history, l)
        if verbose && (length(loss_history) == 1 || length(loss_history) % 100 == 0)
            println(stderr, "NeuralPDE Training: epoch=$(length(loss_history)) loss=$(round(l, sigdigits=4))")
        end
        return false
    end

    res = Optimization.solve(prob, opt; maxiters=config.max_epochs, callback=callback)

    verbose && println(stderr,
        "NeuralPDE Training Complete: epochs=$(length(loss_history)) final_loss=$(round(loss_history[end], sigdigits=4))")

    # Build callable closure for evaluation
    phi_func = discretization.phi[1]
    trained_params = res.u
    eval_closure = let phi_func = phi_func, trained_params = trained_params
        (t_val, x_val, y_val) -> first(phi_func([t_val, x_val, y_val], trained_params))
    end

    return PINNSolution(eval_closure, trained_params, nothing, config, loss_history, domain_nt, grid_ic)
end

end # module
