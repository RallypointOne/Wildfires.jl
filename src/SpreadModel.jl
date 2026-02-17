module SpreadModel

export AbstractWind, AbstractMoisture, AbstractTerrain
export UniformWind, UniformMoisture, UniformSlope, FlatTerrain, DynamicMoisture
export FireSpreadModel, spread_rate_field!, simulate!, fire_loss, update!

using ..Rothermel: Rothermel as RothermelModel, FuelClasses, rate_of_spread
using ..LevelSet: LevelSetGrid, xcoords, ycoords, ignite!, advance!, reinitialize!

#-----------------------------------------------------------------------------# Abstract Types
"""
    AbstractWind

Supertype for wind field components.

Subtypes must be callable as `wind(t, x, y) -> (speed, direction)` where
`speed` is midflame wind speed [km/h] and `direction` is the direction the
wind blows FROM [radians].
"""
abstract type AbstractWind end

"""
    AbstractMoisture

Supertype for fuel moisture components.

Subtypes must be callable as `moisture(t, x, y) -> FuelClasses`.
"""
abstract type AbstractMoisture end

"""
    AbstractTerrain

Supertype for terrain/topography components.

Subtypes must be callable as `terrain(t, x, y) -> (slope, aspect)` where
`slope` is rise/run [fraction] and `aspect` is the downslope direction [radians].
"""
abstract type AbstractTerrain end

#-----------------------------------------------------------------------------# update!
"""
    update!(component, grid::LevelSetGrid, dt)

Update a dynamic component based on the current fire state. Called by `simulate!`
between time steps. Default is a no-op for static components.
"""
update!(::AbstractWind, grid::LevelSetGrid, dt) = nothing
update!(::AbstractMoisture, grid::LevelSetGrid, dt) = nothing
update!(::AbstractTerrain, grid::LevelSetGrid, dt) = nothing

#-----------------------------------------------------------------------------# Wind Components
"""
    UniformWind{T}(; speed, direction=0.0)

Spatially and temporally constant wind field.

# Fields
- `speed::T` - Midflame wind speed [km/h]
- `direction::T` - Direction wind blows FROM [radians]
"""
struct UniformWind{T} <: AbstractWind
    speed::T
    direction::T
end
UniformWind(; speed, direction=0.0) = UniformWind(promote(speed, direction)...)

(w::UniformWind)(t, x, y) = (w.speed, w.direction)

#-----------------------------------------------------------------------------# Moisture Components
"""
    UniformMoisture{T}(moisture::FuelClasses{T})

Spatially and temporally constant fuel moisture.
"""
struct UniformMoisture{T} <: AbstractMoisture
    moisture::FuelClasses{T}
end

(m::UniformMoisture)(t, x, y) = m.moisture

"""
    DynamicMoisture(grid::LevelSetGrid, moisture::FuelClasses; dry_rate=0.1, recovery_rate=0.001, min_d1=0.03)

Spatially varying fuel moisture that responds to fire-induced drying.

The 1-hr dead fuel moisture (`d1`) varies spatially while other size classes remain
constant. As the fire front approaches, radiative heat flux dries unburned fuel
ahead of the front. Moisture also recovers toward the ambient value over time.

The drying model at each unburned cell:

    dM/dt = -dry_rate / (φ² + 1) + recovery_rate · (M_ambient - M)

where `φ` is the level set value (approximate distance to the fire front in meters).
Moisture is clamped to `[min_d1, ambient_d1]` to prevent unrealistic drying.

# Fields
- `d1::Matrix{T}` - Spatially varying 1-hr dead fuel moisture [fraction]
- `base::FuelClasses{T}` - Moisture values for other size classes
- `ambient_d1::T` - Equilibrium d1 moisture from weather [fraction]
- `dry_rate::T` - Fire-induced drying coefficient [fraction/min]
- `recovery_rate::T` - Moisture recovery rate toward ambient [1/min]
- `min_d1::T` - Minimum d1 moisture floor [fraction]
- `dx::T`, `dy::T`, `x0::T`, `y0::T` - Grid geometry for coordinate lookup
"""
mutable struct DynamicMoisture{T} <: AbstractMoisture
    d1::Matrix{T}
    base::FuelClasses{T}
    ambient_d1::T
    dry_rate::T
    recovery_rate::T
    min_d1::T
    dx::T
    dy::T
    x0::T
    y0::T
end

function DynamicMoisture(grid::LevelSetGrid{T}, moisture::FuelClasses{T};
                         dry_rate=T(0.1), recovery_rate=T(0.001), min_d1=T(0.03)) where {T}
    d1 = fill(moisture.d1, size(grid))
    DynamicMoisture(d1, moisture, moisture.d1, T(dry_rate), T(recovery_rate),
                    T(min_d1), grid.dx, grid.dy, grid.x0, grid.y0)
end

function (m::DynamicMoisture{T})(t, x, y) where {T}
    j = clamp(round(Int, (x - m.x0) / m.dx + T(0.5)), 1, size(m.d1, 2))
    i = clamp(round(Int, (y - m.y0) / m.dy + T(0.5)), 1, size(m.d1, 1))
    FuelClasses(d1=m.d1[i, j], d10=m.base.d10, d100=m.base.d100,
                herb=m.base.herb, wood=m.base.wood)
end

function update!(m::DynamicMoisture, grid::LevelSetGrid, dt)
    φ = grid.φ
    for j in axes(φ, 2), i in axes(φ, 1)
        if φ[i, j] > 0  # unburned
            # Fire-induced drying: radiative flux decays with distance²
            # φ ≈ signed distance to front [m] after reinitialization
            fire_flux = m.dry_rate / (φ[i, j]^2 + 1.0)

            # Recovery toward ambient (weather-driven rewetting)
            recovery = m.recovery_rate * (m.ambient_d1 - m.d1[i, j])

            m.d1[i, j] = clamp(m.d1[i, j] + (-fire_flux + recovery) * dt, m.min_d1, m.ambient_d1)
        end
    end
end

#-----------------------------------------------------------------------------# Terrain Components
"""
    FlatTerrain()

Flat terrain (zero slope everywhere).
"""
struct FlatTerrain <: AbstractTerrain end

(::FlatTerrain)(t, x, y) = (zero(x), zero(x))

"""
    UniformSlope{T}(; slope, aspect=0.0)

Spatially constant terrain slope.

# Fields
- `slope::T` - Terrain slope as rise/run [fraction]
- `aspect::T` - Downslope direction [radians]
"""
struct UniformSlope{T} <: AbstractTerrain
    slope::T
    aspect::T
end
UniformSlope(; slope, aspect=0.0) = UniformSlope(promote(slope, aspect)...)

(s::UniformSlope)(t, x, y) = (s.slope, s.aspect)

#-----------------------------------------------------------------------------# FireSpreadModel
"""
    FireSpreadModel(fuel, wind, moisture, terrain)

Composable fire spread model that combines a fuel model with spatially varying
environmental inputs. Callable as `model(t, x, y)` → spread rate [m/min].

Each component is a callable with signature `(t, x, y)`:
- `wind::AbstractWind` → `(speed, direction)`
- `moisture::AbstractMoisture` → `FuelClasses`
- `terrain::AbstractTerrain` → `(slope, aspect)`

Dynamic components (e.g. `DynamicMoisture`) are updated between time steps
via `update!(component, grid, dt)` during `simulate!`.

### Examples
```julia
using Wildfires.Rothermel
using Wildfires.SpreadModel

model = FireSpreadModel(
    SHORT_GRASS,
    UniformWind(speed=8.0),
    UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
    FlatTerrain()
)

model(0.0, 100.0, 100.0)  # spread rate at (t=0, x=100, y=100)
```
"""
struct FireSpreadModel{F,W<:AbstractWind,M<:AbstractMoisture,T<:AbstractTerrain}
    fuel::F
    wind::W
    moisture::M
    terrain::T
end

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
        F[i, j] = model(t, xs[j], ys[i])
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

        if grad > 0
            F[i, j] = _directional_rate(
                model, t, xs[j], ys[i],
                dφdx / grad, dφdy / grad)
        else
            F[i, j] = model(t, xs[j], ys[i])
        end
    end
    F
end

# Direction-dependent spread rate using front normal (nx, ny)
function _directional_rate(model::FireSpreadModel,
        t, x, y, nx, ny)
    speed, wind_dir = model.wind(t, x, y)
    moist = model.moisture(t, x, y)
    slope_val, aspect = model.terrain(t, x, y)

    R_head = rate_of_spread(model.fuel,
        moisture=moist, wind=speed, slope=slope_val)
    R_head == 0 && return 0.0

    R_base = rate_of_spread(model.fuel,
        moisture=moist, wind=0.0, slope=0.0)
    R_head ≈ R_base && return R_head

    # Push direction weighted by each component's
    # contribution to spread rate
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
    pmag == 0 && return R_head

    cos_theta = (nx * px + ny * py) / pmag
    return R_base + (R_head - R_base) * max(0.0, cos_theta)
end

#-----------------------------------------------------------------------------# simulate!
"""
    simulate!(grid::LevelSetGrid, model; steps=100, dt=0.5, reinit_every=10)

Run the level set simulation using a `FireSpreadModel` to compute spread rates.

Between time steps, dynamic components are updated via `update!(model, grid, dt)`.
This allows components like `DynamicMoisture` to respond to the evolving fire state.

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 50.0)
simulate!(grid, model, steps=100, dt=0.5)
```
"""
function simulate!(grid::LevelSetGrid, model; steps::Int=100, dt=0.5, reinit_every::Int=10)
    F = Matrix{eltype(grid)}(undef, size(grid))
    for step in 1:steps
        update!(model, grid, dt)
        spread_rate_field!(F, model, grid)
        advance!(grid, F, dt)
        step % reinit_every == 0 && reinitialize!(grid)
    end
    grid
end

#-----------------------------------------------------------------------------# fire_loss
"""
    fire_loss(grid::LevelSetGrid, φ_observed::AbstractMatrix)

Compute the sum-of-squares loss between the current level set field and an observed field.
"""
fire_loss(grid::LevelSetGrid, φ_observed::AbstractMatrix) = sum(x -> x^2, grid.φ .- φ_observed)

end # module
