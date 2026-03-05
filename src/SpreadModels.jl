module SpreadModels

export AbstractWind, AbstractMoisture, AbstractTerrain, AbstractFuel
export AbstractSpotting, AbstractSuppression, AbstractBurnout, AbstractBurnin
export NoBurnout, ExponentialBurnout, LinearBurnout
export NoBurnin, ExponentialBurnin, LinearBurnin
export UniformWind, UniformMoisture, UniformSlope, FlatTerrain, DynamicMoisture
export RothermelModel, spread_rate_field!, simulate!, fire_loss, update!, directional_speed
export AbstractBlendingMode, CosineBlending, EllipticalBlending
export length_to_breadth, fire_eccentricity
export AbstractSolver, Godunov, Superbee, WENO5
export AbstractReinitMethod, IterativeReinit, NewtonReinit
export Trace

using ..Rothermel: FuelClasses, rate_of_spread
using ..LevelSet: LevelSetGrid, xcoords, ycoords, ignite!, advance!, reinitialize!, cfl_dt,
    AbstractSolver, Godunov, Superbee, WENO5,
    AbstractReinitMethod, IterativeReinit, NewtonReinit,
    _curvature, _phi_safe
using ..Components: AbstractWind, AbstractMoisture, AbstractTerrain, AbstractFuel,
    AbstractSpotting, AbstractSuppression, AbstractBurnout, AbstractBurnin,
    NoBurnout, ExponentialBurnout, LinearBurnout,
    NoBurnin, ExponentialBurnin, LinearBurnin,
    UniformWind, UniformMoisture, FlatTerrain, UniformSlope, DynamicMoisture,
    AbstractBlendingMode, CosineBlending, EllipticalBlending,
    length_to_breadth, fire_eccentricity
import ..Components: update!
import ..CellularAutomata
using ..CellularAutomata: CAGrid, CellState, UNBURNED, BURNING, BURNED, UNBURNABLE,
    AbstractNeighborhood, Moore, VonNeumann, _offsets,
    cellstate, on_ignite, on_burnout

#--------------------------------------------------------------------------------# Fuel Resolution
_resolve_fuel(fuel, t, x, y) = fuel
_resolve_fuel(fuel::AbstractFuel, t, x, y) = fuel(t, x, y)

#--------------------------------------------------------------------------------# RothermelModel
"""
    RothermelModel(fuel, wind, moisture, terrain, [directional])

Composable fire spread model that combines a fuel model with spatially varying
environmental inputs. Callable as `model(t, x, y)` → spread rate [m/min].

Each component is a callable with signature `(t, x, y)`:
- `wind::AbstractWind` → `(speed, direction)`
- `moisture::AbstractMoisture` → `FuelClasses`
- `terrain::AbstractTerrain` → `(slope, aspect)`
- `directional::AbstractBlendingMode` → how spread varies with angle (default: `CosineBlending()`)

Dynamic components (e.g. `DynamicMoisture`) are updated between time steps
via `update!(component, grid, dt)` during `simulate!`.

### Examples
```julia
using Wildfires.Rothermel
using Wildfires.SpreadModels

# Cosine blending (default)
model = RothermelModel(
    SHORT_GRASS,
    UniformWind(speed=8.0),
    UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
    FlatTerrain()
)

# Elliptical blending (more realistic fire shapes)
model = RothermelModel(
    SHORT_GRASS,
    UniformWind(speed=8.0),
    UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
    FlatTerrain(),
    EllipticalBlending(),
)

model(0.0, 100.0, 100.0)  # spread rate at (t=0, x=100, y=100)
```
"""
struct RothermelModel{F,W<:AbstractWind,M<:AbstractMoisture,T<:AbstractTerrain,D<:AbstractBlendingMode}
    fuel::F
    wind::W
    moisture::M
    terrain::T
    directional::D
end
RothermelModel(fuel, wind, moisture, terrain) = RothermelModel(fuel, wind, moisture, terrain, CosineBlending())

function (s::RothermelModel)(t, x, y)
    fuel = _resolve_fuel(s.fuel, t, x, y)
    speed, _ = s.wind(t, x, y)
    moist = s.moisture(t, x, y)
    slope, _ = s.terrain(t, x, y)
    rate_of_spread(fuel, moisture=moist, wind=speed, slope=slope)
end

function update!(model::RothermelModel, grid::LevelSetGrid, dt)
    model.fuel isa AbstractFuel && update!(model.fuel, grid, dt)
    update!(model.wind, grid, dt)
    update!(model.moisture, grid, dt)
    update!(model.terrain, grid, dt)
end

#-----------------------------------------------------------------------------# directional_speed
"""
    directional_speed(model::RothermelModel, t, x, y, nx, ny)

Compute the direction-dependent spread rate [m/min] for the given fire propagation
direction `(nx, ny)`.

`(nx, ny)` is the unit normal of the fire front (same convention as `∇φ/|∇φ|` in
[`spread_rate_field!`](@ref)).  The spread rate varies with the angle between the
propagation direction and the combined wind/slope push direction, using the model's
`directional` blending strategy ([`CosineBlending`](@ref) or [`EllipticalBlending`](@ref)).

### Examples
```julia
model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
directional_speed(model, 0.0, 100.0, 100.0, 1.0, 0.0)  # speed in +x direction
```
"""
function directional_speed(model::RothermelModel, t, x, y, nx, ny)
    _directional_rate(model, model.directional, t, x, y, nx, ny)
end

#-----------------------------------------------------------------------------# spread_rate_field!
"""
    spread_rate_field!(F::AbstractMatrix, model, grid::LevelSetGrid)

Fill matrix `F` by evaluating `model(t, x, y)` at each cell center of `grid`.

For a `RothermelModel`, the spread rate is direction-dependent: the Rothermel
head-fire rate applies in the wind/slope push direction, while the base rate
(no wind, no slope) applies for flanking and backing fire.  The effective rate
at each cell is:

    F = R_base + (R_head - R_base) · max(0, n̂ · d̂)

where `n̂ = ∇φ/|∇φ|` is the front propagation direction and `d̂` is the
combined wind + slope push direction (weighted by their individual
contributions to spread rate).
"""
function spread_rate_field!(F::AbstractMatrix, model, grid::LevelSetGrid)
    xs = xcoords(grid)
    ys = ycoords(grid)
    t = grid.t
    for j in eachindex(xs), i in eachindex(ys)
        F[i, j] = isnan(grid.t_ignite[i, j]) ? zero(eltype(F)) : model(t, xs[j], ys[i])
    end
    F
end

function spread_rate_field!(F::AbstractMatrix, model::RothermelModel, grid::LevelSetGrid)
    xs = xcoords(grid)
    ys = ycoords(grid)
    t = grid.t
    φ = grid.φ
    nrows, ncols = size(φ)
    dxg, dyg = grid.dx, grid.dy

    for j in eachindex(xs), i in eachindex(ys)
        # Front normal from ∇φ (central differences)
        dφdx = if j == 1
            (φ[i, 2] - φ[i, 1]) / dxg
        elseif j == ncols
            (φ[i, ncols] - φ[i, ncols-1]) / dxg
        else
            (φ[i, j+1] - φ[i, j-1]) / (2dxg)
        end
        dφdy = if i == 1
            (φ[2, j] - φ[1, j]) / dyg
        elseif i == nrows
            (φ[nrows, j] - φ[nrows-1, j]) / dyg
        else
            (φ[i+1, j] - φ[i-1, j]) / (2dyg)
        end
        grad = hypot(dφdx, dφdy)

        if isnan(grid.t_ignite[i, j])
            F[i, j] = zero(eltype(F))
        elseif grad > 0
            F[i, j] = _directional_rate(
                model, model.directional, t, xs[j], ys[i],
                dφdx / grad, dφdy / grad)
        else
            F[i, j] = model(t, xs[j], ys[i])
        end
    end
    F
end

# Shared helper: compute push direction and cos(theta) with front normal
function _push_direction(model::RothermelModel, t, x, y, nx, ny)
    fuel = _resolve_fuel(model.fuel, t, x, y)
    speed, wind_dir = model.wind(t, x, y)
    moist = model.moisture(t, x, y)
    slope_val, aspect = model.terrain(t, x, y)

    R_head = rate_of_spread(fuel,
        moisture=moist, wind=speed, slope=slope_val)
    R_base = rate_of_spread(fuel,
        moisture=moist, wind=0.0, slope=0.0)

    # Push direction weighted by each component's contribution to spread rate
    R_w = rate_of_spread(fuel,
        moisture=moist, wind=speed, slope=0.0)
    R_s = rate_of_spread(fuel,
        moisture=moist, wind=0.0, slope=slope_val)
    w_wind = R_w - R_base
    w_slope = R_s - R_base

    # Wind pushes opposite to FROM direction
    # Slope pushes uphill (opposite to aspect)
    px = w_wind * (-cos(wind_dir)) +
         w_slope * (-cos(aspect))
    py = w_wind * (-sin(wind_dir)) +
         w_slope * (-sin(aspect))
    pmag = hypot(px, py)

    cos_theta = pmag == 0 ? one(px) : (nx * px + ny * py) / pmag
    return (; R_head, R_base, speed, cos_theta, fuel, moist)
end

# Cosine blending: R = R_base + (R_head - R_base) * max(0, cos θ)
function _directional_rate(model::RothermelModel, ::CosineBlending,
        t, x, y, nx, ny)
    (; R_head, R_base, cos_theta) = _push_direction(model, t, x, y, nx, ny)
    R_head == 0 && return 0.0
    R_head ≈ R_base && return R_head
    return R_base + (R_head - R_base) * max(0.0, cos_theta)
end

# Find wind speed (km/h) that alone produces R_target, capturing combined wind+slope effect.
# Uses bisection on rate_of_spread (monotonic in wind).
function _effective_wind_speed(fuel, moist, R_target, R_base)
    R_target <= R_base && return 0.0
    lo, hi = 0.0, 50.0
    while rate_of_spread(fuel, moisture=moist, wind=hi, slope=0.0) < R_target
        hi *= 2
    end
    for _ in 1:20
        mid = (lo + hi) / 2
        R_mid = rate_of_spread(fuel, moisture=moist, wind=mid, slope=0.0)
        if R_mid < R_target
            lo = mid
        else
            hi = mid
        end
        (hi - lo) < 0.1 && break
    end
    return (lo + hi) / 2
end

# Elliptical blending: correct normal speed for elliptical fire shape
#
# The fire ellipse (Anderson 1983) has the ignition at the rear focus.
# Its evolution decomposes into:
#   - Expansion: the ellipse grows with semi-axes a (head) and b=a/LB (flank)
#   - Drift: the center moves in the push direction at speed a·ε
#
# The corresponding normal speed is:
#   F_n = R_head/(1+ε) · (sqrt(cos²θ + sin²θ/LB²) + ε·cos θ)
#
# This gives R_head at the head (θ=0) and R_head·(1-ε)/(1+ε) at the
# backing (θ=π), with smooth elliptical flanks.
function _directional_rate(model::RothermelModel, dir::EllipticalBlending,
        t, x, y, nx, ny)
    (; R_head, R_base, cos_theta, fuel, moist) = _push_direction(model, t, x, y, nx, ny)
    R_head == 0 && return 0.0
    R_head ≈ R_base && return R_head

    # Effective wind speed accounts for both wind and slope
    U_eff_kmh = _effective_wind_speed(fuel, moist, R_head, R_base)
    U_eff_ms = U_eff_kmh / 3.6
    LB = length_to_breadth(U_eff_ms; formula=dir.formula)
    ε = fire_eccentricity(LB)

    sin2 = 1 - cos_theta^2
    R_expand = R_head / (1 + ε)
    F_n = R_expand * (sqrt(cos_theta^2 + sin2 / LB^2) + ε * cos_theta)
    return max(F_n, R_base)
end

#-----------------------------------------------------------------------------# Trace
"""
    Trace{T, M}

Records snapshots of grid state during [`simulate!`](@ref).

Works with both `LevelSetGrid` (snapshots of `φ`) and `CAGrid` (snapshots of cell states).

# Fields
- `stack::Vector{Tuple{T, M}}` - Collected `(time, snapshot)` pairs
- `every::Int` - Record every N simulation steps

# Constructors
    Trace(grid::LevelSetGrid, every::Integer)
    Trace(grid::CAGrid, every::Integer)

### Examples
```julia
# With LevelSetGrid
grid = LevelSetGrid(100, 100, dx=30.0)
trace = Trace(grid, 5)
simulate!(grid, model, steps=100, trace=trace)

# With CAGrid
grid = CAGrid(100, 100, dx=30.0)
trace = Trace(grid, 5)
simulate!(grid, model, steps=100, trace=trace)
```
"""
struct Trace{T, M}
    stack::Vector{Tuple{T, M}}
    every::Int
end

function Trace(grid::LevelSetGrid{T}, every::Integer) where {T}
    Trace{T, Matrix{T}}(Tuple{T, Matrix{T}}[(grid.t, collect(grid.φ))], every)
end

function Trace(grid::CAGrid{T, S}, every::Integer) where {T, S}
    Trace{T, Matrix{S}}(Tuple{T, Matrix{S}}[(grid.t, copy(grid.state))], every)
end

"""
    CATrace

Deprecated alias for [`Trace`](@ref). Use `Trace(grid::CAGrid, every)` instead.
"""
const CATrace = Trace

function _record!(trace::Trace, grid::LevelSetGrid)
    push!(trace.stack, (grid.t, collect(grid.φ)))
end

function _record!(trace::Trace, grid::CAGrid)
    push!(trace.stack, (grid.t, copy(grid.state)))
end

#-----------------------------------------------------------------------------# simulate!
"""
    simulate!(grid::LevelSetGrid, model; steps=100, dt=nothing, cfl=0.5, reinit_every=10, burnout=nothing, burnin=nothing, trace=nothing, progress=false, solver=Godunov(), curvature=0.0, reinit=IterativeReinit())

Run the level set simulation using a `RothermelModel` to compute spread rates.

When `dt` is `nothing` (the default), the time step is computed automatically each
step via the CFL condition: `dt = cfl * min(dx, dy) / max(F)`.  Pass an explicit
`dt` to use a fixed time step instead.

Between time steps, dynamic components are updated via `update!(model, grid, dt)`.
This allows components like `DynamicMoisture` to respond to the evolving fire state.

# Burnout

Pass `burnout` as an [`AbstractBurnout`](@ref) model to scale spread rates based on burn
duration.  Available models: [`NoBurnout`](@ref) (default, no scaling),
[`ExponentialBurnout(τ)`](@ref) (exponential decay), [`LinearBurnout(τ)`](@ref) (linear ramp).

For backward compatibility, passing a `Real` value coerces to `ExponentialBurnout(val)` and
`nothing` coerces to `NoBurnout()`.

# Burn-in

Pass `burnin` as an [`AbstractBurnin`](@ref) model to ramp up spread rates after ignition.
This prevents freshly ignited cells from immediately propagating fire in all directions.
Available models: [`NoBurnin`](@ref) (default, instant full intensity),
[`ExponentialBurnin(τ)`](@ref) (exponential ramp-up), [`LinearBurnin(τ)`](@ref) (linear ramp-up).

The total scaling at each cell is `burnin(t) * burnout(t)`.

# Curvature Regularization

Pass `curvature > 0` to penalize high curvature of the fire front.  The effective
spread rate becomes `max(F - b·κ, 0)` where `b` is the curvature coefficient and
`κ` is the mean curvature at each cell.  This smooths the fire front by reducing
speed at convex bulges and increasing it at concave indentations.

# Reinitialization

Pass `reinit` as an [`AbstractReinitMethod`](@ref) to select the reinitialization
algorithm: [`IterativeReinit`](@ref) (default) or [`NewtonReinit`](@ref).

# Trace

Pass a [`Trace`](@ref) to record snapshots of `φ` at regular intervals for animation:

    trace = Trace(grid, 5)
    simulate!(grid, model, steps=100, trace=trace)

# Progress

Pass `progress=true` to display a progress meter during simulation.

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 50.0)

# Automatic CFL-limited time stepping (recommended)
simulate!(grid, model, steps=100)

# Fixed time step (user must ensure CFL stability)
simulate!(grid, model, steps=100, dt=0.5)

# With burnout (exponential decay)
simulate!(grid, model, steps=100, burnout=ExponentialBurnout(residence_time(SHORT_GRASS)))

# With burn-in (prevents instant propagation from freshly ignited cells)
simulate!(grid, model, steps=100, burnin=ExponentialBurnin(0.5))

# With curvature regularization
simulate!(grid, model, steps=100, curvature=5.0)

# With Newton reinitialization
simulate!(grid, model, steps=100, reinit=NewtonReinit())

# With trace for animation
trace = Trace(grid, 10)
simulate!(grid, model, steps=100, trace=trace)

# With progress meter
simulate!(grid, model, steps=1000, progress=true)
```
"""
function simulate!(grid::LevelSetGrid, model; steps::Int=100, dt=nothing, cfl=0.5, reinit_every::Int=10, burnout=nothing, burnin=nothing, trace=nothing, progress::Bool=false, solver::AbstractSolver=Godunov(), curvature=0.0, reinit::AbstractReinitMethod=IterativeReinit())
    bo = _coerce_burnout(burnout)
    bi = _coerce_burnin(burnin)
    F = similar(grid.φ)
    for step in 1:steps
        spread_rate_field!(F, model, grid)
        _scale_burnout!(F, grid, bo)
        _scale_burnin!(F, grid, bi)
        curvature > 0 && _apply_curvature!(F, grid, curvature)
        step_dt = dt === nothing ? cfl_dt(grid, F; cfl=cfl, curvature=curvature) : dt
        update!(model, grid, step_dt)
        advance!(grid, F, step_dt, solver)
        step % reinit_every == 0 && reinitialize!(grid, reinit)
        trace !== nothing && step % trace.every == 0 && _record!(trace, grid)
        progress && step % max(1, steps ÷ 100) == 0 && _print_progress(step, steps, grid)
    end
    progress && println()
    grid
end

function _print_progress(step, steps, grid)
    pct = round(Int, 100 * step / steps)
    n_burned = count(<(0), grid.φ)
    n_total = length(grid.φ)
    print("\r  step $step/$steps ($pct%) | t = $(round(grid.t, digits=2)) min | burned = $n_burned/$n_total")
end

#-----------------------------------------------------------------------------# Burnout coercion and scaling
_coerce_burnout(::Nothing) = NoBurnout()
_coerce_burnout(t_r::Real) = ExponentialBurnout(t_r)
_coerce_burnout(bo::AbstractBurnout) = bo

_scale_burnout!(F, grid, ::NoBurnout) = nothing

function _scale_burnout!(F, grid, bo::AbstractBurnout)
    for j in axes(F, 2), i in axes(F, 1)
        t_ig = grid.t_ignite[i, j]
        if isfinite(t_ig)
            F[i, j] *= bo(grid.t - t_ig)
        end
    end
end

#-----------------------------------------------------------------------------# Burn-in coercion and scaling
_coerce_burnin(::Nothing) = NoBurnin()
_coerce_burnin(bi::AbstractBurnin) = bi

_scale_burnin!(F, grid, ::NoBurnin) = nothing

function _scale_burnin!(F, grid, bi::AbstractBurnin)
    for j in axes(F, 2), i in axes(F, 1)
        t_ig = grid.t_ignite[i, j]
        if isfinite(t_ig)
            F[i, j] *= bi(grid.t - t_ig)
        end
    end
end

#-----------------------------------------------------------------------------# Curvature regularization
function _apply_curvature!(F, grid, b)
    φ = grid.φ
    ny, nx = size(φ)
    dx, dy = grid.dx, grid.dy
    bc = grid.bc
    z = zero(eltype(F))
    for j in 1:nx, i in 1:ny
        κ = _curvature(φ, i, j, ny, nx, dx, dy, bc)
        F[i, j] = max(F[i, j] - b * κ, z)
    end
end

#-----------------------------------------------------------------------------# fire_loss
"""
    fire_loss(grid::LevelSetGrid, φ_observed::AbstractMatrix)

Compute the sum-of-squares loss between the current level set field and an observed field:

``\\sum_i (\\phi_i - \\phi^{\\text{obs}}_i)^2``

This is useful for calibrating model parameters (wind speed, moisture, etc.) against
observed fire perimeter data, or as an objective function in optimization/inverse problems.

### Examples
```julia
# Compare simulation result against observed fire state
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 50.0)
simulate!(grid, model, steps=100)

# φ_observed from satellite/sensor data (same grid dimensions)
loss = fire_loss(grid, φ_observed)
```
"""
fire_loss(grid::LevelSetGrid, φ_observed::AbstractMatrix) = sum(x -> x^2, grid.φ .- φ_observed)

#==============================================================================#
#                     Cellular Automata (Travel-Time)                          #
#==============================================================================#

#-----------------------------------------------------------------------------# update! for CAGrid
function update!(model::RothermelModel, grid::CAGrid, dt)
    model.fuel isa AbstractFuel && update!(model.fuel, grid, dt)
    update!(model.wind, grid, dt)
    update!(model.moisture, grid, dt)
    update!(model.terrain, grid, dt)
end

update!(::AbstractWind, ::CAGrid, dt) = nothing
update!(::AbstractMoisture, ::CAGrid, dt) = nothing
update!(::AbstractTerrain, ::CAGrid, dt) = nothing
update!(::AbstractFuel, ::CAGrid, dt) = nothing

function update!(::DynamicMoisture, ::CAGrid, dt)
    error("DynamicMoisture requires LevelSetGrid (phi field). Use UniformMoisture with CAGrid.")
end


#-----------------------------------------------------------------------------# CFL for CAGrid
function _ca_cfl_dt(grid::CAGrid{T}, model; cfl=0.5) where {T}
    xs = CellularAutomata.xcoords(grid)
    ys = CellularAutomata.ycoords(grid)
    R_max = zero(T)
    for j in eachindex(xs), i in eachindex(ys)
        cellstate(grid.state[i, j]) == BURNING || continue
        R = model(grid.t, xs[j], ys[i])
        R_max = max(R_max, R)
    end
    R_max > 0 || return T(Inf)
    return T(cfl) * min(grid.dx, grid.dy) / R_max
end

#-----------------------------------------------------------------------------# advance! for CAGrid
"""
    advance!(grid::CAGrid, model::RothermelModel, dt; burnout=NoBurnout(), burnin=NoBurnin(), residence_time=Inf)

Advance the CA fire simulation by one time step `dt` [min] using a deterministic
travel-time approach.

For each `BURNING` cell, the directional spread rate `R(θ)` is computed to each
`UNBURNED` neighbor using [`directional_speed`](@ref).  The travel time is
`distance / R(θ)`, and the neighbor ignites when its minimum arrival time is
reached.  `BURNING` cells transition to `BURNED` after `residence_time` minutes.

### Examples
```julia
grid = CAGrid(50, 50, dx=30.0)
ignite!(grid, 750.0, 750.0, 60.0)
model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
advance!(grid, model, 0.5)
```
"""
function CellularAutomata.advance!(g::CAGrid{T}, model::RothermelModel, dt;
        burnout::AbstractBurnout=NoBurnout(),
        burnin::AbstractBurnin=NoBurnin(),
        residence_time=T(Inf)) where {T}
    state = g.state
    ny, nx = size(state)
    dx, dy = g.dx, g.dy
    offsets = _offsets(g.neighborhood)
    xs = CellularAutomata.xcoords(g)
    ys = CellularAutomata.ycoords(g)
    t_now = g.t + dt

    # Phase 1: Compute arrival times from BURNING -> UNBURNED neighbors
    for j in 1:nx, i in 1:ny
        cellstate(state[i, j]) == BURNING || continue

        t_burn = g.t - g.t_ignite[i, j]
        scale = burnout(t_burn) * burnin(t_burn)
        scale > zero(T) || continue

        x_src = xs[j]
        y_src = ys[i]

        for (di, dj) in offsets
            ni, nj = i + di, j + dj
            (1 <= ni <= ny && 1 <= nj <= nx) || continue
            cellstate(state[ni, nj]) == UNBURNED || continue

            delta_x = T(dj) * dx
            delta_y = T(di) * dy
            dist = hypot(delta_x, delta_y)
            norm_x = delta_x / dist
            norm_y = delta_y / dist

            R = directional_speed(model, g.t, x_src, y_src, norm_x, norm_y) * scale
            R > zero(T) || continue

            t_arr = g.t_ignite[i, j] + dist / R
            g.t_arrival[ni, nj] = min(g.t_arrival[ni, nj], t_arr)
        end
    end

    # Phase 2: Ignite cells whose arrival time <= t_now
    for j in 1:nx, i in 1:ny
        if cellstate(state[i, j]) == UNBURNED && g.t_arrival[i, j] <= t_now
            state[i, j] = on_ignite(state[i, j], g.t_arrival[i, j])
            g.t_ignite[i, j] = g.t_arrival[i, j]
            g.t_arrival[i, j] = T(Inf)
        end
    end

    # Phase 3: Burnout (BURNING -> BURNED)
    for j in 1:nx, i in 1:ny
        if cellstate(state[i, j]) == BURNING
            elapsed = t_now - g.t_ignite[i, j]
            if elapsed >= residence_time
                state[i, j] = on_burnout(state[i, j], t_now)
            end
        end
    end

    g.t = t_now
    g
end

#-----------------------------------------------------------------------------# simulate! for CAGrid
"""
    simulate!(grid::CAGrid, model::RothermelModel; steps=100, dt=nothing, cfl=0.5, burnout=nothing, burnin=nothing, residence_time=Inf, trace=nothing, progress=false)

Run a deterministic cellular automata fire simulation using a travel-time approach.

The same `RothermelModel` used with `LevelSetGrid` works here. The CA uses
[`directional_speed`](@ref) to compute direction-dependent spread rates to each
neighbor, supporting both [`CosineBlending`](@ref) and [`EllipticalBlending`](@ref).

When `dt` is `nothing` (the default), the time step is computed automatically each
step via the CFL condition: `dt = cfl * min(dx, dy) / max(R)`.

# Burnout

Cells transition from `BURNING` to `BURNED` after `residence_time` minutes (default
`Inf` = burn forever).  Additionally, pass `burnout` as an [`AbstractBurnout`](@ref)
to scale outgoing spread rates based on burn duration.

# Trace

Pass a [`Trace`](@ref) to record state snapshots at regular intervals:

    trace = Trace(grid, 5)
    simulate!(grid, model, steps=100, trace=trace)

### Examples
```julia
grid = CAGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)

model = RothermelModel(
    SHORT_GRASS,
    UniformWind(speed=8.0),
    UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
    FlatTerrain(),
    EllipticalBlending(),
)

# Automatic CFL-limited time stepping
simulate!(grid, model, steps=200)

# With residence time (cells burn out after 5 min)
simulate!(grid, model, steps=200, residence_time=5.0)

# With trace for animation
trace = Trace(grid, 10)
simulate!(grid, model, steps=200, trace=trace)
```
"""
function simulate!(grid::CAGrid{T}, model::RothermelModel;
        steps::Int=100, dt=nothing, cfl=0.5,
        burnout=nothing, burnin=nothing,
        residence_time=T(Inf),
        trace=nothing, progress::Bool=false) where {T}
    bo = _coerce_burnout(burnout)
    bi = _coerce_burnin(burnin)
    for step in 1:steps
        step_dt = dt === nothing ? _ca_cfl_dt(grid, model; cfl=cfl) : dt
        update!(model, grid, step_dt)
        CellularAutomata.advance!(grid, model, step_dt;
            burnout=bo, burnin=bi, residence_time=residence_time)
        trace !== nothing && step % trace.every == 0 && _record!(trace, grid)
        progress && step % max(1, steps ÷ 100) == 0 && _print_ca_progress(step, steps, grid)
    end
    progress && println()
    grid
end

function _print_ca_progress(step, steps, grid)
    pct = round(Int, 100 * step / steps)
    n_burning = count(c -> cellstate(c) == BURNING, grid.state)
    n_burned = count(grid.state) do c
        s = cellstate(c)
        s == BURNING || s == BURNED
    end
    n_total = length(grid.state)
    print("\r  step $step/$steps ($pct%) | t = $(round(grid.t, digits=2)) min | burned = $n_burned/$n_total | burning = $n_burning")
end

end # module
