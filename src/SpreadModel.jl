module SpreadModel

export AbstractWind, AbstractMoisture, AbstractTerrain
export AbstractSpotting, AbstractSuppression, AbstractBurnout, AbstractBurnin
export NoBurnout, ExponentialBurnout, LinearBurnout
export NoBurnin, ExponentialBurnin, LinearBurnin
export UniformWind, UniformMoisture, UniformSlope, FlatTerrain, DynamicMoisture
export FireSpreadModel, spread_rate_field!, simulate!, fire_loss, update!
export AbstractDirectionalModel, CosineBlending, EllipticalBlending
export length_to_breadth, fire_eccentricity
export Trace

using ..Rothermel: Rothermel as RothermelModel, FuelClasses, rate_of_spread
using ..LevelSet: LevelSetGrid, xcoords, ycoords, ignite!, advance!, reinitialize!, cfl_dt
using ..Components: AbstractWind, AbstractMoisture, AbstractTerrain,
    AbstractSpotting, AbstractSuppression, AbstractBurnout, AbstractBurnin,
    NoBurnout, ExponentialBurnout, LinearBurnout,
    NoBurnin, ExponentialBurnin, LinearBurnin,
    UniformWind, UniformMoisture, FlatTerrain, UniformSlope, DynamicMoisture,
    AbstractDirectionalModel, CosineBlending, EllipticalBlending,
    length_to_breadth, fire_eccentricity
import ..Components: update!

#--------------------------------------------------------------------------------# FireSpreadModel
"""
    FireSpreadModel(fuel, wind, moisture, terrain, [directional])

Composable fire spread model that combines a fuel model with spatially varying
environmental inputs. Callable as `model(t, x, y)` → spread rate [m/min].

Each component is a callable with signature `(t, x, y)`:
- `wind::AbstractWind` → `(speed, direction)`
- `moisture::AbstractMoisture` → `FuelClasses`
- `terrain::AbstractTerrain` → `(slope, aspect)`
- `directional::AbstractDirectionalModel` → how spread varies with angle (default: `CosineBlending()`)

Dynamic components (e.g. `DynamicMoisture`) are updated between time steps
via `update!(component, grid, dt)` during `simulate!`.

### Examples
```julia
using Wildfires.Rothermel
using Wildfires.SpreadModel

# Cosine blending (default)
model = FireSpreadModel(
    SHORT_GRASS,
    UniformWind(speed=8.0),
    UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
    FlatTerrain()
)

# Elliptical blending (more realistic fire shapes)
model = FireSpreadModel(
    SHORT_GRASS,
    UniformWind(speed=8.0),
    UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
    FlatTerrain(),
    EllipticalBlending(),
)

model(0.0, 100.0, 100.0)  # spread rate at (t=0, x=100, y=100)
```
"""
struct FireSpreadModel{F,W<:AbstractWind,M<:AbstractMoisture,T<:AbstractTerrain,D<:AbstractDirectionalModel}
    fuel::F
    wind::W
    moisture::M
    terrain::T
    directional::D
end
FireSpreadModel(fuel, wind, moisture, terrain) = FireSpreadModel(fuel, wind, moisture, terrain, CosineBlending())

function (s::FireSpreadModel)(t, x, y)
    speed, _ = s.wind(t, x, y)
    moist = s.moisture(t, x, y)
    slope, _ = s.terrain(t, x, y)
    rate_of_spread(s.fuel, moisture=moist, wind=speed, slope=slope)
end

function update!(model::FireSpreadModel, grid::LevelSetGrid, dt)
    update!(model.wind, grid, dt)
    update!(model.moisture, grid, dt)
    update!(model.terrain, grid, dt)
end

#-----------------------------------------------------------------------------# spread_rate_field!
"""
    spread_rate_field!(F::AbstractMatrix, model, grid::LevelSetGrid)

Fill matrix `F` by evaluating `model(t, x, y)` at each cell center of `grid`.

For a `FireSpreadModel`, the spread rate is direction-dependent: the Rothermel
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

function spread_rate_field!(F::AbstractMatrix, model::FireSpreadModel, grid::LevelSetGrid)
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
function _push_direction(model::FireSpreadModel, t, x, y, nx, ny)
    speed, wind_dir = model.wind(t, x, y)
    moist = model.moisture(t, x, y)
    slope_val, aspect = model.terrain(t, x, y)

    R_head = rate_of_spread(model.fuel,
        moisture=moist, wind=speed, slope=slope_val)
    R_base = rate_of_spread(model.fuel,
        moisture=moist, wind=0.0, slope=0.0)

    # Push direction weighted by each component's contribution to spread rate
    R_w = rate_of_spread(model.fuel,
        moisture=moist, wind=speed, slope=0.0)
    R_s = rate_of_spread(model.fuel,
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
    return (; R_head, R_base, speed, cos_theta)
end

# Cosine blending: R = R_base + (R_head - R_base) * max(0, cos θ)
function _directional_rate(model::FireSpreadModel, ::CosineBlending,
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
function _directional_rate(model::FireSpreadModel, dir::EllipticalBlending,
        t, x, y, nx, ny)
    (; R_head, R_base, cos_theta) = _push_direction(model, t, x, y, nx, ny)
    R_head == 0 && return 0.0
    R_head ≈ R_base && return R_head

    # Effective wind speed accounts for both wind and slope
    moist = model.moisture(t, x, y)
    U_eff_kmh = _effective_wind_speed(model.fuel, moist, R_head, R_base)
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
    Trace{T}

Records snapshots of the level set field during [`simulate!`](@ref).

# Fields
- `stack::Vector{Tuple{T, Matrix{T}}}` - Collected `(time, φ)` snapshots
- `every::Int` - Record every N simulation steps

# Constructor
    Trace(grid::LevelSetGrid, every::Integer)

Create a trace that records the initial grid state and will record every `every` steps
during `simulate!`.

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 50.0)
trace = Trace(grid, 5)
simulate!(grid, model, steps=100, trace=trace)

length(trace.stack)  # 1 (initial) + 20 (every 5 of 100 steps) = 21
trace.stack[1]       # (0.0, Matrix{Float64})
```
"""
struct Trace{T}
    stack::Vector{Tuple{T, Matrix{T}}}
    every::Int
end

function Trace(grid::LevelSetGrid{T}, every::Integer) where {T}
    Trace{T}(Tuple{T, Matrix{T}}[(grid.t, collect(grid.φ))], every)
end

function _record!(trace::Trace, grid::LevelSetGrid)
    push!(trace.stack, (grid.t, collect(grid.φ)))
end

#-----------------------------------------------------------------------------# simulate!
"""
    simulate!(grid::LevelSetGrid, model; steps=100, dt=nothing, cfl=0.5, reinit_every=10, burnout=nothing, burnin=nothing, trace=nothing, progress=false)

Run the level set simulation using a `FireSpreadModel` to compute spread rates.

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

# With trace for animation
trace = Trace(grid, 10)
simulate!(grid, model, steps=100, trace=trace)

# With progress meter
simulate!(grid, model, steps=1000, progress=true)
```
"""
function simulate!(grid::LevelSetGrid, model; steps::Int=100, dt=nothing, cfl=0.5, reinit_every::Int=10, burnout=nothing, burnin=nothing, trace=nothing, progress::Bool=false)
    bo = _coerce_burnout(burnout)
    bi = _coerce_burnin(burnin)
    F = similar(grid.φ)
    for step in 1:steps
        spread_rate_field!(F, model, grid)
        _scale_burnout!(F, grid, bo)
        _scale_burnin!(F, grid, bi)
        step_dt = dt === nothing ? cfl_dt(grid, F; cfl=cfl) : dt
        update!(model, grid, step_dt)
        advance!(grid, F, step_dt)
        step % reinit_every == 0 && reinitialize!(grid)
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

end # module
