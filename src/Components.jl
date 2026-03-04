module Components

using ..Rothermel: FuelClasses
using ..LevelSet: LevelSetGrid

export AbstractWind, AbstractMoisture, AbstractTerrain
export AbstractSpotting, AbstractSuppression, AbstractBurnout, AbstractBurnin
export NoBurnout, ExponentialBurnout, LinearBurnout
export NoBurnin, ExponentialBurnin, LinearBurnin
export UniformWind, UniformMoisture, FlatTerrain, UniformSlope, DynamicMoisture
export AbstractBlendingMode, CosineBlending, EllipticalBlending
export length_to_breadth, fire_eccentricity
export update!

#-----------------------------------------------------------------------------# Abstract Types
"""
    AbstractWind

Supertype for wind field components.

## Interface

Subtypes must be callable as `wind(t, x, y) -> (speed, direction)` where:
- `speed` — midflame wind speed [km/h]
- `direction` — direction the wind blows FROM [radians]

Optionally implement [`update!`](@ref)`(wind, grid, dt)` for time-varying fields.

Concrete types: [`UniformWind`](@ref).
"""
abstract type AbstractWind end

"""
    AbstractMoisture

Supertype for fuel moisture components.

## Interface

Subtypes must be callable as `moisture(t, x, y) -> FuelClasses`.

Optionally implement [`update!`](@ref)`(moisture, grid, dt)` for fire-responsive moisture.

Concrete types: [`UniformMoisture`](@ref), [`DynamicMoisture`](@ref).
"""
abstract type AbstractMoisture end

"""
    AbstractTerrain

Supertype for terrain/topography components.

## Interface

Subtypes must be callable as `terrain(t, x, y) -> (slope, aspect)` where:
- `slope` — rise/run [fraction]
- `aspect` — downslope direction [radians]

Optionally implement [`update!`](@ref)`(terrain, grid, dt)` for dynamic terrain.

Concrete types: [`FlatTerrain`](@ref), [`UniformSlope`](@ref).
"""
abstract type AbstractTerrain end

"""
    AbstractSpotting

Supertype for ember transport / spot fire ignition components.

## Interface

Subtypes must implement `spot!(grid, model, dt)` which checks for lofted firebrands
and ignites new spot fires ahead of the main front.

!!! note
    Reserved for future use — not yet connected to `simulate!`.
"""
abstract type AbstractSpotting end

"""
    AbstractSuppression

Supertype for fire suppression components (fireline construction, retardant drops, etc.).

## Interface

Subtypes must implement `suppress!(grid, model, dt)` which modifies the grid to
reflect suppression actions (e.g., creating unburnable barriers or reducing spread rates).

!!! note
    Reserved for future use — not yet connected to `simulate!`.
"""
abstract type AbstractSuppression end

"""
    AbstractBurnout

Supertype for burnout models that scale fire spread from exhausted fuel.

Subtypes must be callable as `burnout(t_burning) -> Float64 ∈ [0, 1]` where
`t_burning` is the time [min] since ignition.  Returns `1.0` for full intensity
and `0.0` for fully burned out (fuel exhausted, no longer contributing to spread).

Concrete types: [`NoBurnout`](@ref), [`ExponentialBurnout`](@ref), [`LinearBurnout`](@ref).
"""
abstract type AbstractBurnout end

"""
    NoBurnout()

No burnout — fire spreads at full intensity indefinitely once ignited.
"""
struct NoBurnout <: AbstractBurnout end
(::NoBurnout)(t_burning) = one(t_burning)

"""
    ExponentialBurnout(τ)

Exponential decay burnout model.  The spread intensity decays as `exp(-t_burning / τ)`
where `τ` is the residence time [min].

### Examples
```julia
b = ExponentialBurnout(0.005)
b(0.0)    # 1.0  (just ignited)
b(0.005)  # ≈ 0.37  (one residence time)
b(0.02)   # ≈ 0.02  (nearly exhausted)
```
"""
struct ExponentialBurnout{T} <: AbstractBurnout
    τ::T
end
(b::ExponentialBurnout)(t_burning) = exp(-t_burning / b.τ)

"""
    LinearBurnout(τ)

Linear decay burnout model.  The spread intensity decreases linearly from `1` to `0`
over residence time `τ` [min], then remains at zero.

### Examples
```julia
b = LinearBurnout(10.0)
b(0.0)   # 1.0
b(5.0)   # 0.5
b(10.0)  # 0.0
b(15.0)  # 0.0
```
"""
struct LinearBurnout{T} <: AbstractBurnout
    τ::T
end
(b::LinearBurnout)(t_burning) = max(zero(t_burning), one(t_burning) - t_burning / b.τ)

#-----------------------------------------------------------------------------# Burn-in Components
"""
    AbstractBurnin

Supertype for burn-in models that ramp up fire spread intensity after ignition.

Subtypes must be callable as `burnin(t_burning) -> Float64 ∈ [0, 1]` where
`t_burning` is the time [min] since ignition.  Returns `0.0` at ignition
(no spread contribution) and ramps up to `1.0` (full intensity) as the fire
establishes in the cell.

This prevents freshly ignited cells from immediately propagating fire in all
directions — a cell that just ignited on one side needs time before it can
spread to the opposite side.

Concrete types: [`NoBurnin`](@ref), [`ExponentialBurnin`](@ref), [`LinearBurnin`](@ref).
"""
abstract type AbstractBurnin end

"""
    NoBurnin()

No burn-in delay — fire spreads at full intensity immediately upon ignition.
"""
struct NoBurnin <: AbstractBurnin end
(::NoBurnin)(t_burning) = one(t_burning)

"""
    ExponentialBurnin(τ)

Exponential ramp-up burn-in model.  The spread intensity ramps as `1 - exp(-t_burning / τ)`
where `τ` is the characteristic ignition establishment time [min].

### Examples
```julia
b = ExponentialBurnin(0.5)
b(0.0)   # 0.0  (just ignited, no spread)
b(0.5)   # ≈ 0.63  (one time constant)
b(2.0)   # ≈ 0.98  (nearly full intensity)
```
"""
struct ExponentialBurnin{T} <: AbstractBurnin
    τ::T
end
(b::ExponentialBurnin)(t_burning) = one(t_burning) - exp(-t_burning / b.τ)

"""
    LinearBurnin(τ)

Linear ramp-up burn-in model.  The spread intensity increases linearly from `0` to `1`
over time `τ` [min], then remains at full intensity.

### Examples
```julia
b = LinearBurnin(1.0)
b(0.0)   # 0.0
b(0.5)   # 0.5
b(1.0)   # 1.0
b(2.0)   # 1.0
```
"""
struct LinearBurnin{T} <: AbstractBurnin
    τ::T
end
(b::LinearBurnin)(t_burning) = min(one(t_burning), t_burning / b.τ)

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

#-----------------------------------------------------------------------------# update!
"""
    update!(component, grid::LevelSetGrid, dt)

Update a dynamic component based on the current fire state. Called by `simulate!`
between time steps. Default is a no-op for static components.
"""
function update! end
update!(::AbstractWind, grid::LevelSetGrid, dt) = nothing
update!(::AbstractMoisture, grid::LevelSetGrid, dt) = nothing
update!(::AbstractTerrain, grid::LevelSetGrid, dt) = nothing

#-----------------------------------------------------------------------------# DynamicMoisture
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
- `d1::M` - Spatially varying 1-hr dead fuel moisture (`M <: AbstractMatrix{T}`) [fraction]
- `base::FuelClasses{T}` - Moisture values for other size classes
- `ambient_d1::T` - Equilibrium d1 moisture from weather [fraction]
- `dry_rate::T` - Fire-induced drying coefficient [fraction/min]
- `recovery_rate::T` - Moisture recovery rate toward ambient [1/min]
- `min_d1::T` - Minimum d1 moisture floor [fraction]
- `dx::T`, `dy::T`, `x0::T`, `y0::T` - Grid geometry for coordinate lookup
"""
mutable struct DynamicMoisture{T, M <: AbstractMatrix{T}} <: AbstractMoisture
    d1::M
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
            fire_flux = m.dry_rate / (φ[i, j]^2 + 1)

            # Recovery toward ambient (weather-driven rewetting)
            recovery = m.recovery_rate * (m.ambient_d1 - m.d1[i, j])

            m.d1[i, j] = clamp(m.d1[i, j] + (-fire_flux + recovery) * dt, m.min_d1, m.ambient_d1)
        end
    end
end

#--------------------------------------------------------------------------------# Directional Spread Models
"""
    AbstractBlendingMode

Supertype for directional fire spread models that control how the spread rate
varies with angle relative to the dominant push direction (wind + slope).

"Blending" refers to the interpolation scheme that computes the spread rate at
an arbitrary angle `θ` from the push direction, given the head-fire rate
`R_head` (maximum, aligned with push) and a base rate `R_base` (no wind/slope).

- [`CosineBlending`](@ref) — uses a cosine weighting: `R(θ) = R_base + (R_head - R_base) · max(0, cos θ)`.
  Simple but produces fires wider than typically observed.
- [`EllipticalBlending`](@ref) — uses an elliptical fire shape (Anderson 1983) parameterized
  by the length-to-breadth ratio and eccentricity derived from wind speed. Produces
  the narrow, elongated fire shapes seen in real fires under wind.
"""
abstract type AbstractBlendingMode end

"""
    CosineBlending()

Cosine-based directional spread model (default).

The spread rate at angle `θ` from the push direction is:

    R(θ) = R_base + (R_head - R_base) · max(0, cos θ)

This produces fires that are somewhat wider than observed in practice.
"""
struct CosineBlending <: AbstractBlendingMode end

"""
    EllipticalBlending(; formula=:anderson)

Elliptical fire spread model based on Anderson (1983).

The normal speed at angle `θ` from the push direction is:

    F_n(θ) = R_head/(1+ε) · (√(cos²θ + sin²θ/LB²) + ε·cos θ)

where `ε` is the fire eccentricity and `LB` the length-to-breadth ratio,
both derived from the effective midflame wind speed.  The first term is the
ellipse expansion and the second is the drift that places the ignition at
the rear focus (as in the Anderson/Richards fire ellipse convention).

# Fields
- `formula::Symbol` - Length-to-breadth formula: `:anderson` (default) or `:green`

### Examples
```julia
model = RothermelModel(
    SHORT_GRASS,
    UniformWind(speed=8.0),
    UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
    FlatTerrain(),
    EllipticalBlending(),
)
```
"""
struct EllipticalBlending <: AbstractBlendingMode
    formula::Symbol
end
EllipticalBlending(; formula::Symbol=:anderson) = EllipticalBlending(formula)

#--------------------------------------------------------------------------------# Length-to-Breadth and Eccentricity
"""
    length_to_breadth(U; formula=:anderson)

Compute the fire length-to-breadth ratio from effective midflame wind speed `U` [m/s].

# Formulas
- `:anderson` (default) — Anderson (1983):
  `LB = 0.936 · exp(0.2566 · U) + 0.461 · exp(-0.1548 · U) - 0.397`
- `:green` — Green (1983): `LB = 1.1 · U^0.464` (suitable for grass)

Returns at least `1.0` (a circle at zero wind).

### Examples
```julia
length_to_breadth(0.0)   # ≈ 1.0 (circle)
length_to_breadth(2.0)   # > 1.0 (elongated)
```
"""
function length_to_breadth(U; formula::Symbol=:anderson)
    if formula === :anderson
        LB = oftype(U, 0.936) * exp(oftype(U, 0.2566) * U) +
             oftype(U, 0.461) * exp(oftype(U, -0.1548) * U) -
             oftype(U, 0.397)
    elseif formula === :green
        LB = oftype(U, 1.1) * U^oftype(U, 0.464)
    else
        error("Unknown LB formula: $formula. Use :anderson or :green.")
    end
    return max(LB, one(LB))
end

"""
    fire_eccentricity(LB)

Compute fire eccentricity from the length-to-breadth ratio `LB`.

Returns `0` for `LB = 1` (circle), approaching `1` for large `LB` (highly elongated).

### Examples
```julia
fire_eccentricity(1.0)  # 0.0 (circle)
fire_eccentricity(3.0)  # ≈ 0.943
```
"""
fire_eccentricity(LB) = sqrt(LB^2 - one(LB)) / LB

end # module
