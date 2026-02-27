module WildfiresPINNExt

using Wildfires
using Wildfires: PINNConfig, PINNSolution
using Wildfires.LevelSet: LevelSetGrid, xcoords, ycoords

using Lux
using ComponentArrays: ComponentArray
using ForwardDiff
using Zygote
using Optimization
using OptimizationOptimisers

import Random

#--------------------------------------------------------------------------------# Network Construction
function build_network(config::PINNConfig)
    act = _get_activation(config.activation)
    dims = config.hidden_dims
    layers = []
    push!(layers, Dense(3 => dims[1], act))
    for i in 2:length(dims)
        push!(layers, Dense(dims[i-1] => dims[i], act))
    end
    push!(layers, Dense(dims[end] => 1))
    Chain(layers...)
end

function _get_activation(s::Symbol)
    s === :tanh && return tanh
    s === :relu && return relu
    s === :sigmoid && return sigmoid
    s === :gelu && return gelu
    error("Unknown activation: $s. Use :tanh, :relu, :sigmoid, or :gelu.")
end

#--------------------------------------------------------------------------------# Input Normalization
function normalize_input(t, x, y, domain)
    t_min, t_max = domain.tspan
    x_min, x_max = domain.xspan
    y_min, y_max = domain.yspan
    t_norm = 2 * (t - t_min) / (t_max - t_min) - 1
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
    return (t_norm, x_norm, y_norm)
end

#--------------------------------------------------------------------------------# Bilinear Interpolation (plain function, never inside AD)
"""
    bilinear_interp(φ, x0_grid, y0_grid, dx, dy, x, y)

Bilinear interpolation of grid values `φ` at physical coordinates `(x, y)`.
Grid is described by origin `(x0_grid, y0_grid)` and spacing `(dx, dy)`.
"""
function bilinear_interp(φ::Matrix, x0_grid::Float64, y0_grid::Float64, dx::Float64, dy::Float64, x::Float64, y::Float64)
    ny, nx = size(φ)
    fx = clamp((x - x0_grid) / dx, 0.0, Float64(nx - 1))
    fy = clamp((y - y0_grid) / dy, 0.0, Float64(ny - 1))
    j0 = clamp(floor(Int, fx) + 1, 1, nx - 1)
    i0 = clamp(floor(Int, fy) + 1, 1, ny - 1)
    j1 = min(j0 + 1, nx)
    i1 = min(i0 + 1, ny)
    tx = fx - (j0 - 1)
    ty = fy - (i0 - 1)
    return (1 - tx) * (1 - ty) * φ[i0, j0] +
           tx * (1 - ty) * φ[i0, j1] +
           (1 - tx) * ty * φ[i1, j0] +
           tx * ty * φ[i1, j1]
end

"""
    bilinear_interp_grad(φ, x0_grid, y0_grid, dx, dy, x, y) → (∂f/∂x, ∂f/∂y)

Analytical spatial gradient of bilinear interpolation.
"""
function bilinear_interp_grad(φ::Matrix, x0_grid::Float64, y0_grid::Float64, dx::Float64, dy::Float64, x::Float64, y::Float64)
    ny, nx = size(φ)
    fx = clamp((x - x0_grid) / dx, 0.0, Float64(nx - 1))
    fy = clamp((y - y0_grid) / dy, 0.0, Float64(ny - 1))
    j0 = clamp(floor(Int, fx) + 1, 1, nx - 1)
    i0 = clamp(floor(Int, fy) + 1, 1, ny - 1)
    j1 = min(j0 + 1, nx)
    i1 = min(i0 + 1, ny)
    tx = fx - (j0 - 1)
    ty = fy - (i0 - 1)
    dfdx = ((1 - ty) * (φ[i0, j1] - φ[i0, j0]) + ty * (φ[i1, j1] - φ[i1, j0])) / dx
    dfdy = ((1 - tx) * (φ[i1, j0] - φ[i0, j0]) + tx * (φ[i1, j1] - φ[i0, j1])) / dy
    return (dfdx, dfdy)
end

#--------------------------------------------------------------------------------# Full Solution: IC + NN correction
# Hard IC constraint: φ̃(x,y,t) = IC(x,y)/L + τ(t) · NN(x_n, y_n, t_n)
# where τ(t) = (t - t_min) / (t_max - t_min), so φ̃(x,y,t_min) = IC(x,y)/L exactly.

function eval_phi_tilde(model, ps, st, t, x, y, domain, ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, phi_scale)
    t_min, t_max = domain.tspan
    ic_val = Zygote.ignore_derivatives() do
        bilinear_interp(ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, x, y) / phi_scale
    end
    tau = (t - t_min) / (t_max - t_min)
    t_n, x_n, y_n = normalize_input(t, x, y, domain)
    nn_input = [x_n, y_n, t_n]
    nn_out, _ = model(nn_input, ps, st)
    return ic_val + tau * nn_out[1]
end

#--------------------------------------------------------------------------------# Derivative Computation (analytical, no bilinear_interp in AD)
# φ̃(x,y,t) = IC(x,y)/L + τ(t) · NN(x_n(x), y_n(y), t_n(t))
#
# ∂φ̃/∂t = τ'(t) · NN + τ(t) · ∂NN/∂t_n · dt_n/dt
# ∂φ̃/∂x = ∂IC/∂x / L + τ(t) · ∂NN/∂x_n · dx_n/dx
# ∂φ̃/∂y = ∂IC/∂y / L + τ(t) · ∂NN/∂y_n · dy_n/dy

function compute_derivatives(model, ps, st, t, x, y, domain, ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, norm_consts)
    (; dx_n_dx, dy_n_dy, dt_n_dt, dtau_dt, inv_phi_scale) = norm_consts
    t_min, _ = domain.tspan

    # IC spatial gradient (analytical, completely outside AD)
    dic_dx, dic_dy = Zygote.ignore_derivatives() do
        bilinear_interp_grad(ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, x, y)
    end
    dic_dx *= inv_phi_scale
    dic_dy *= inv_phi_scale

    # Time factor
    tau = (t - t_min) * dtau_dt

    # NN value and gradient via ForwardDiff (only the NN model, no bilinear_interp)
    t_n, x_n, y_n = normalize_input(t, x, y, domain)
    function nn_scalar(v)
        out, _ = model(v, ps, st)
        return out[1]
    end
    nn_input = [x_n, y_n, t_n]
    nn_val = nn_scalar(nn_input)
    nn_grad = ForwardDiff.gradient(nn_scalar, nn_input)  # [∂NN/∂x_n, ∂NN/∂y_n, ∂NN/∂t_n]

    # Full derivatives (using precomputed normalization constants)
    dphi_dt = dtau_dt * nn_val + tau * nn_grad[3] * dt_n_dt
    dphi_dx = dic_dx + tau * nn_grad[1] * dx_n_dx
    dphi_dy = dic_dy + tau * nn_grad[2] * dy_n_dy

    return (dphi_dt, dphi_dx, dphi_dy)
end

#--------------------------------------------------------------------------------# Collocation Point Sampling
function sample_collocation(config::PINNConfig, domain, rng; ic_φ=nothing, ic_x0=0.0, ic_y0=0.0, ic_dx=1.0, ic_dy=1.0)
    t_min, t_max = domain.tspan
    x_min, x_max = domain.xspan
    y_min, y_max = domain.yspan

    n_int = config.n_interior

    if config.importance_sampling && ic_φ !== nothing
        # Sample half uniformly, half concentrated near fire front (|IC| small)
        n_uniform = n_int ÷ 2
        n_front = n_int - n_uniform

        # Uniform samples
        t_uni = t_min .+ (t_max - t_min) .* rand(rng, n_uniform)
        x_uni = x_min .+ (x_max - x_min) .* rand(rng, n_uniform)
        y_uni = y_min .+ (y_max - y_min) .* rand(rng, n_uniform)

        # Front-concentrated: oversample and keep those with small |IC|
        n_candidates = n_front * 10
        x_cand = x_min .+ (x_max - x_min) .* rand(rng, n_candidates)
        y_cand = y_min .+ (y_max - y_min) .* rand(rng, n_candidates)
        ic_vals = [abs(bilinear_interp(ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, x_cand[i], y_cand[i])) for i in 1:n_candidates]

        # Weight inversely by |IC| + small offset, then resample
        weights = 1.0 ./ (ic_vals .+ 1e-2 * maximum(abs, ic_φ))
        weights ./= sum(weights)

        # Systematic resampling
        cum_weights = cumsum(weights)
        indices = Vector{Int}(undef, n_front)
        u = rand(rng) / n_front
        j = 1
        for i in 1:n_front
            target = u + (i - 1) / n_front
            while j < n_candidates && cum_weights[j] < target
                j += 1
            end
            indices[i] = j
        end

        x_front = x_cand[indices]
        y_front = y_cand[indices]
        t_front = t_min .+ (t_max - t_min) .* rand(rng, n_front)

        t_int = vcat(t_uni, t_front)
        x_int = vcat(x_uni, x_front)
        y_int = vcat(y_uni, y_front)
    else
        t_int = t_min .+ (t_max - t_min) .* rand(rng, n_int)
        x_int = x_min .+ (x_max - x_min) .* rand(rng, n_int)
        y_int = y_min .+ (y_max - y_min) .* rand(rng, n_int)
    end

    # BC points (on domain boundary, random t)
    n_per_side = config.n_boundary ÷ 4
    t_bc = t_min .+ (t_max - t_min) .* rand(rng, 4 * n_per_side)

    x_bc = Float64[]
    y_bc = Float64[]

    # Left (x = x_min)
    append!(x_bc, fill(x_min, n_per_side))
    append!(y_bc, y_min .+ (y_max - y_min) .* rand(rng, n_per_side))
    # Right (x = x_max)
    append!(x_bc, fill(x_max, n_per_side))
    append!(y_bc, y_min .+ (y_max - y_min) .* rand(rng, n_per_side))
    # Bottom (y = y_min)
    append!(x_bc, x_min .+ (x_max - x_min) .* rand(rng, n_per_side))
    append!(y_bc, fill(y_min, n_per_side))
    # Top (y = y_max)
    append!(x_bc, x_min .+ (x_max - x_min) .* rand(rng, n_per_side))
    append!(y_bc, fill(y_max, n_per_side))

    return (
        t_int=t_int, x_int=x_int, y_int=y_int,
        t_bc=t_bc, x_bc=x_bc, y_bc=y_bc,
    )
end

#--------------------------------------------------------------------------------# Loss Function
function pinn_loss(θ, p)
    (; model, st, config, domain, spread_model, ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, colloc, observations, phi_scale, norm_consts) = p
    (; dtau_dt, inv_phi_scale) = norm_consts

    t_min, _ = domain.tspan
    ε = 1e-6

    # --- PDE residual loss: ∂φ̃/∂t + F|∇φ̃| = 0 (per-point ForwardDiff.gradient) ---
    loss_pde = zero(eltype(θ))
    n_int = length(colloc.t_int)
    for k in 1:n_int
        t_k, x_k, y_k = colloc.t_int[k], colloc.x_int[k], colloc.y_int[k]
        dφ_dt, dφ_dx, dφ_dy = compute_derivatives(model, θ, st, t_k, x_k, y_k, domain, ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, norm_consts)
        grad_mag = sqrt(dφ_dx^2 + dφ_dy^2 + ε^2)
        F = Zygote.ignore_derivatives() do
            spread_model(t_k, x_k, y_k)
        end
        residual = dφ_dt + F * grad_mag
        loss_pde += residual^2
    end
    loss_pde /= n_int

    # --- BC loss (batched forward pass) ---
    n_bc = length(colloc.x_bc)
    loss_bc = zero(eltype(θ))
    if n_bc > 0
        # Build normalized input matrix (3, N) for batched NN eval
        input_bc = Matrix{Float64}(undef, 3, n_bc)
        tau_bc = Vector{Float64}(undef, n_bc)
        ic_bc_vals = Vector{Float64}(undef, n_bc)
        Zygote.ignore_derivatives() do
            for k in 1:n_bc
                t_n, x_n, y_n = normalize_input(colloc.t_bc[k], colloc.x_bc[k], colloc.y_bc[k], domain)
                input_bc[1, k] = x_n
                input_bc[2, k] = y_n
                input_bc[3, k] = t_n
                tau_bc[k] = (colloc.t_bc[k] - t_min) * dtau_dt
                ic_bc_vals[k] = bilinear_interp(ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, colloc.x_bc[k], colloc.y_bc[k]) * inv_phi_scale
            end
        end
        nn_out_bc, _ = model(input_bc, θ, st)  # (1, n_bc) batched forward pass
        for k in 1:n_bc
            φ_pred = ic_bc_vals[k] + tau_bc[k] * nn_out_bc[1, k]
            loss_bc += max(zero(eltype(θ)), -φ_pred)^2
        end
        loss_bc /= n_bc
    end

    # --- Data loss (batched forward pass) ---
    loss_data = zero(eltype(θ))
    if observations !== nothing
        t_obs, x_obs, y_obs, φ_obs = observations
        n_obs = length(t_obs)
        input_obs = Matrix{Float64}(undef, 3, n_obs)
        tau_obs = Vector{Float64}(undef, n_obs)
        ic_obs_vals = Vector{Float64}(undef, n_obs)
        Zygote.ignore_derivatives() do
            for k in 1:n_obs
                t_n, x_n, y_n = normalize_input(t_obs[k], x_obs[k], y_obs[k], domain)
                input_obs[1, k] = x_n
                input_obs[2, k] = y_n
                input_obs[3, k] = t_n
                tau_obs[k] = (t_obs[k] - t_min) * dtau_dt
                ic_obs_vals[k] = bilinear_interp(ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, x_obs[k], y_obs[k]) * inv_phi_scale
            end
        end
        nn_out_obs, _ = model(input_obs, θ, st)  # (1, n_obs) batched forward pass
        for k in 1:n_obs
            φ_pred = ic_obs_vals[k] + tau_obs[k] * nn_out_obs[1, k]
            loss_data += (φ_pred - φ_obs[k] * inv_phi_scale)^2
        end
        loss_data /= n_obs
    end

    return config.lambda_pde * loss_pde +
           config.lambda_bc * loss_bc +
           config.lambda_data * loss_data
end

#--------------------------------------------------------------------------------# train_pinn
function Wildfires.train_pinn(grid::LevelSetGrid, spread_model, tspan::Tuple,
                              config::PINNConfig;
                              observations=nothing,
                              rng=Random.default_rng(),
                              verbose::Bool=true,
                              lbfgs_optimizer=nothing)
    xs = xcoords(grid)
    ys = ycoords(grid)

    # Output normalization scale: NN correction is O(1), physical φ = φ̃ * phi_scale.
    phi_scale = maximum(abs, grid.φ)
    if phi_scale == 0
        phi_scale = one(eltype(grid.φ))
    end

    domain = (
        tspan = tspan,
        xspan = (first(xs) - step(xs)/2, last(xs) + step(xs)/2),
        yspan = (first(ys) - step(ys)/2, last(ys) + step(ys)/2),
        phi_scale = phi_scale,
    )

    # Precompute normalization constants
    t_min, t_max = domain.tspan
    x_min, x_max = domain.xspan
    y_min, y_max = domain.yspan
    norm_consts = (
        dx_n_dx = 2.0 / (x_max - x_min),
        dy_n_dy = 2.0 / (y_max - y_min),
        dt_n_dt = 2.0 / (t_max - t_min),
        dtau_dt = 1.0 / (t_max - t_min),
        inv_phi_scale = 1.0 / phi_scale,
    )

    # Build network and initialize parameters
    nn = build_network(config)
    if config.float32
        ps, st = Lux.f32(Lux.setup(rng, nn))
    else
        ps, st = Lux.f64(Lux.setup(rng, nn))
    end
    θ = ComponentArray(ps)

    # Snapshot initial condition (grid values and grid geometry as plain Float64)
    ic_φ = copy(grid.φ)
    ic_x0 = Float64(first(xs))
    ic_y0 = Float64(first(ys))
    ic_dx = Float64(step(xs))
    ic_dy = Float64(step(ys))
    grid_ic = LevelSetGrid(copy(grid.φ), grid.dx, grid.dy, grid.x0, grid.y0, grid.t, grid.bc)

    # Collocation points (mutable for resampling)
    colloc_ref = Ref(sample_collocation(config, domain, rng;
        ic_φ=ic_φ, ic_x0=ic_x0, ic_y0=ic_y0, ic_dx=ic_dx, ic_dy=ic_dy))

    loss_history = Float64[]

    p = (
        model = nn,
        st = st,
        config = config,
        domain = domain,
        spread_model = spread_model,
        ic_φ = ic_φ,
        ic_x0 = ic_x0,
        ic_y0 = ic_y0,
        ic_dx = ic_dx,
        ic_dy = ic_dy,
        colloc = colloc_ref[],
        observations = observations,
        phi_scale = phi_scale,
        norm_consts = norm_consts,
    )

    function loss_wrapper(θ_inner, p_inner)
        p_with_colloc = merge(p_inner, (; colloc=colloc_ref[]))
        pinn_loss(θ_inner, p_with_colloc)
    end

    epoch_counter = Ref(0)
    function callback(state, loss_val)
        epoch_counter[] += 1
        push!(loss_history, loss_val)

        if verbose && (epoch_counter[] == 1 || epoch_counter[] % 100 == 0)
            println(stderr, "PINN Training: epoch=$(epoch_counter[]) loss=$(round(loss_val, sigdigits=4))")
        end

        if config.resample_every > 0 && epoch_counter[] % config.resample_every == 0
            colloc_ref[] = sample_collocation(config, domain, rng;
                ic_φ=ic_φ, ic_x0=ic_x0, ic_y0=ic_y0, ic_dx=ic_dx, ic_dy=ic_dy)
            verbose && println(stderr, "Resampled collocation points: epoch=$(epoch_counter[])")
        end

        return false
    end

    # Suppress known benign Zygote warning about ForwardDiff.gradient closures.
    prev_logger = Base.CoreLogging.global_logger()
    Base.CoreLogging.global_logger(Base.CoreLogging.SimpleLogger(stderr, Base.CoreLogging.Error))

    local result
    try
        optf = OptimizationFunction(loss_wrapper, Optimization.AutoZygote())
        optprob = OptimizationProblem(optf, θ, p)
        result = solve(optprob, OptimizationOptimisers.Adam(config.learning_rate);
                       maxiters=config.max_epochs, callback=callback)

        # L-BFGS refinement phase
        if config.lbfgs_epochs > 0
            lbfgs_optimizer === nothing && error("lbfgs_epochs > 0 requires passing an optimizer, e.g. `lbfgs_optimizer=OptimizationOptimJL.LBFGS()`")
            verbose && println(stderr, "PINN: Starting L-BFGS refinement ($(config.lbfgs_epochs) iterations)...")
            optprob_lbfgs = OptimizationProblem(optf, result.u, p)
            result = solve(optprob_lbfgs, lbfgs_optimizer;
                           maxiters=config.lbfgs_epochs, callback=callback)
        end
    finally
        Base.CoreLogging.global_logger(prev_logger)
    end

    verbose && println(stderr, "PINN Training Complete: epochs=$(epoch_counter[]) final_loss=$(round(loss_history[end], sigdigits=4))")

    # Store a callable closure for post-training inference
    eval_fn = let nn = nn, ps = result.u, st = st, domain = domain,
                  ic_φ = ic_φ, ic_x0 = ic_x0, ic_y0 = ic_y0, ic_dx = ic_dx, ic_dy = ic_dy,
                  phi_scale = phi_scale
        function (t, x, y)
            ic_val = bilinear_interp(ic_φ, ic_x0, ic_y0, ic_dx, ic_dy, Float64(x), Float64(y)) / phi_scale
            t_min, t_max = domain.tspan
            tau = (t - t_min) / (t_max - t_min)
            t_n, x_n, y_n = normalize_input(t, x, y, domain)
            nn_out, _ = nn([x_n, y_n, t_n], ps, st)
            (ic_val + tau * nn_out[1]) * phi_scale
        end
    end

    return PINNSolution(eval_fn, result.u, st, config, loss_history, domain, grid_ic)
end

end # module
