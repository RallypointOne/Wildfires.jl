module Propagation

using Extents: Extent
using ..Grids: Grid, xcoords, ycoords
using ..RateOfSpread: AbstractRateOfSpread, rate_of_spread
using ..FireShape: AbstractFireShape, CosineShape, EllipticalShape,
    length_to_breadth, fire_eccentricity, normal_speed, radial_speed
using ..Environment: NoWind, UniformWind, UniformMoisture, GrasslandWeather, ForestWeather,
    FBPWeather, FlatTerrain, UniformSlope, FireEnvironment, head_and_base

# ============================================================================ #
#                              Exports
# ============================================================================ #

# Re-export from FireShape
export AbstractFireShape, CosineShape, EllipticalShape
export length_to_breadth, fire_eccentricity

# Re-export from Environment
export NoWind, UniformWind, UniformMoisture, GrasslandWeather, ForestWeather, FBPWeather
export FlatTerrain, UniformSlope
export FireEnvironment

export CellState, UNBURNED, BURNING, BURNED, UNBURNABLE
export AbstractNeighborhood, Moore, VonNeumann
export AbstractBoundaryCondition, ZeroNeumann, Dirichlet, Periodic
export AbstractSolver, Godunov, Superbee, WENO5
export AbstractReinitMethod, IterativeReinit
export AbstractBurnout, NoBurnout, ExponentialBurnout, LinearBurnout
export AbstractBurnin, NoBurnin, ExponentialBurnin, LinearBurnin
export FireModel
export AbstractPropagationModel, LevelSetPropagation, CAPropagation
export levelset_grid, ca_grid
export ignite!, set_unburnable!
export burned, burning, burnable, burn_area
export advance!, simulate!, cfl_dt, reinitialize!
export spread_rate_field!, directional_speed
export Trace

# ============================================================================ #
#                         Cell States (CA)
# ============================================================================ #

"""
    CellState

Discrete cell states for the cellular automata fire model.

Values: `UNBURNED`, `BURNING`, `BURNED`, `UNBURNABLE`.
"""
@enum CellState::UInt8 UNBURNED BURNING BURNED UNBURNABLE

cellstate(s::CellState) = s

# ============================================================================ #
#                         Neighborhoods (CA)
# ============================================================================ #

abstract type AbstractNeighborhood end

"""
    Moore()

8-connected neighborhood (cardinal + diagonal neighbors).
"""
struct Moore <: AbstractNeighborhood end

"""
    VonNeumann()

4-connected neighborhood (cardinal neighbors only).
"""
struct VonNeumann <: AbstractNeighborhood end

_offsets(::Moore) = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_offsets(::VonNeumann) = ((-1, 0), (0, -1), (0, 1), (1, 0))

# ============================================================================ #
#                       Boundary Conditions
# ============================================================================ #

abstract type AbstractBoundaryCondition end
struct ZeroNeumann <: AbstractBoundaryCondition end
struct Dirichlet <: AbstractBoundaryCondition end
struct Periodic <: AbstractBoundaryCondition end

#-----------------------------------------------------------------------------# Finite differences
@inline _Dxm(φ, i, j, dx, ::ZeroNeumann) = j > 1 ? (φ[i, j] - φ[i, j-1]) / dx : zero(eltype(φ))
@inline _Dxp(φ, i, j, nx, dx, ::ZeroNeumann) = j < nx ? (φ[i, j+1] - φ[i, j]) / dx : zero(eltype(φ))
@inline _Dym(φ, i, j, dx, ::ZeroNeumann) = i > 1 ? (φ[i, j] - φ[i-1, j]) / dx : zero(eltype(φ))
@inline _Dyp(φ, i, j, ny, dx, ::ZeroNeumann) = i < ny ? (φ[i+1, j] - φ[i, j]) / dx : zero(eltype(φ))

@inline _Dxm(φ, i, j, dx, ::Dirichlet) = _Dxm(φ, i, j, dx, ZeroNeumann())
@inline _Dxp(φ, i, j, nx, dx, ::Dirichlet) = _Dxp(φ, i, j, nx, dx, ZeroNeumann())
@inline _Dym(φ, i, j, dx, ::Dirichlet) = _Dym(φ, i, j, dx, ZeroNeumann())
@inline _Dyp(φ, i, j, ny, dx, ::Dirichlet) = _Dyp(φ, i, j, ny, dx, ZeroNeumann())

@inline function _Dxm(φ, i, j, dx, ::Periodic)
    jm = j > 1 ? j - 1 : size(φ, 2)
    (φ[i, j] - φ[i, jm]) / dx
end
@inline function _Dxp(φ, i, j, nx, dx, ::Periodic)
    jp = j < nx ? j + 1 : 1
    (φ[i, jp] - φ[i, j]) / dx
end
@inline function _Dym(φ, i, j, dx, ::Periodic)
    im = i > 1 ? i - 1 : size(φ, 1)
    (φ[i, j] - φ[im, j]) / dx
end
@inline function _Dyp(φ, i, j, ny, dx, ::Periodic)
    ip = i < ny ? i + 1 : 1
    (φ[ip, j] - φ[i, j]) / dx
end

@inline _skip_update(i, j, ny, nx, ::AbstractBoundaryCondition) = false
@inline _skip_update(i, j, ny, nx, ::Dirichlet) = i == 1 || i == ny || j == 1 || j == nx

# ============================================================================ #
#                            Solvers
# ============================================================================ #

abstract type AbstractSolver end

"""
    Godunov()

First-order Godunov upwind scheme with forward Euler time stepping (default).
"""
struct Godunov <: AbstractSolver end

"""
    Superbee(; phi_clamp=100.0, grad_clamp=1000.0)

Second-order Superbee flux limiter with RK2 (Heun's method) time stepping.
"""
struct Superbee{T} <: AbstractSolver
    phi_clamp::T
    grad_clamp::T
end
Superbee(; phi_clamp=100.0, grad_clamp=1000.0) = Superbee(phi_clamp, grad_clamp)

"""
    WENO5(; phi_clamp=100.0)

Fifth-order WENO scheme with SSP-RK3 time stepping.
"""
struct WENO5{T} <: AbstractSolver
    phi_clamp::T
end
WENO5(; phi_clamp=100.0) = WENO5(phi_clamp)

# ============================================================================ #
#                       Reinitialization
# ============================================================================ #

abstract type AbstractReinitMethod end

"""
    IterativeReinit(; iterations=5)

PDE-based iterative reinitialization (Sussman et al. 1994).
"""
struct IterativeReinit <: AbstractReinitMethod
    iterations::Int
end
IterativeReinit(; iterations=5) = IterativeReinit(iterations)

# ============================================================================ #
#                       Burnout / Burnin
# ============================================================================ #

abstract type AbstractBurnout end

"""No burnout — fire spreads at full intensity indefinitely."""
struct NoBurnout <: AbstractBurnout end
(::NoBurnout)(t_burning) = one(t_burning)

"""Exponential decay burnout: `exp(-t/τ)`."""
struct ExponentialBurnout{T} <: AbstractBurnout
    τ::T
end
(b::ExponentialBurnout)(t_burning) = exp(-t_burning / b.τ)

"""Linear decay burnout: `max(0, 1 - t/τ)`."""
struct LinearBurnout{T} <: AbstractBurnout
    τ::T
end
(b::LinearBurnout)(t_burning) = max(zero(t_burning), one(t_burning) - t_burning / b.τ)

abstract type AbstractBurnin end

"""No burn-in delay — full intensity immediately."""
struct NoBurnin <: AbstractBurnin end
(::NoBurnin)(t_burning) = one(t_burning)

"""Exponential ramp-up: `1 - exp(-t/τ)`."""
struct ExponentialBurnin{T} <: AbstractBurnin
    τ::T
end
(b::ExponentialBurnin)(t_burning) = one(t_burning) - exp(-t_burning / b.τ)

"""Linear ramp-up: `min(1, t/τ)`."""
struct LinearBurnin{T} <: AbstractBurnin
    τ::T
end
(b::LinearBurnin)(t_burning) = min(one(t_burning), t_burning / b.τ)

_coerce_burnout(::Nothing) = NoBurnout()
_coerce_burnout(t_r::Real) = ExponentialBurnout(t_r)
_coerce_burnout(bo::AbstractBurnout) = bo

_coerce_burnin(::Nothing) = NoBurnin()
_coerce_burnin(bi::AbstractBurnin) = bi

# ============================================================================ #
#                          FireModel
# ============================================================================ #

"""
    FireModel(ros, shape, env)
    FireModel(ros, wind, weather, terrain, [shape])

Composable fire spread model combining a rate-of-spread model, fire shape,
and environment. Used with [`simulate!`](@ref) to drive fire propagation.

### Examples
```julia
# Full constructor
model = FireModel(
    RothermelROS(SHORT_GRASS),
    EllipticalShape(),
    FireEnvironment(
        UniformWind(speed=8.0),
        UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
        FlatTerrain(),
    ),
)

# Convenience constructor
model = FireModel(
    RothermelROS(SHORT_GRASS),
    UniformWind(speed=8.0),
    UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
    FlatTerrain(),
    EllipticalShape(),
)
```
"""
struct FireModel{R, S <: AbstractFireShape, E}
    ros::R
    shape::S
    env::E
end

FireModel(ros, wind, weather, terrain) =
    FireModel(ros, CosineShape(), FireEnvironment(wind, weather, terrain))
FireModel(ros, wind, weather, terrain, shape) =
    FireModel(ros, shape, FireEnvironment(wind, weather, terrain))

function (fm::FireModel)(t, x, y)
    hb = head_and_base(fm.ros, fm.env, fm.shape, t, x, y)
    hb.R_head
end

# ============================================================================ #
#                    Propagation Models
# ============================================================================ #

"""
    AbstractPropagationModel

Supertype for fire propagation methods. Subtypes control HOW the fire front
is tracked and advanced, while [`FireModel`](@ref) controls WHAT drives it.
"""
abstract type AbstractPropagationModel end

"""
    LevelSetPropagation(; solver=Godunov(), bc=ZeroNeumann(), reinit=IterativeReinit(), reinit_every=10, curvature=0.0)

Level set propagation method. Tracks the fire front as the zero contour of a
signed distance function φ, evolved by `∂φ/∂t + F|∇φ| = 0`.
"""
struct LevelSetPropagation{S <: AbstractSolver, BC <: AbstractBoundaryCondition, R <: AbstractReinitMethod} <: AbstractPropagationModel
    solver::S
    bc::BC
    reinit::R
    reinit_every::Int
    curvature::Float64
end

function LevelSetPropagation(;
        solver::AbstractSolver=Godunov(),
        bc::AbstractBoundaryCondition=ZeroNeumann(),
        reinit::AbstractReinitMethod=IterativeReinit(),
        reinit_every::Int=10,
        curvature::Float64=0.0)
    LevelSetPropagation(solver, bc, reinit, reinit_every, curvature)
end

"""
    CAPropagation(; neighborhood=Moore())

Cellular automata propagation method. Uses a deterministic travel-time approach
where fire spreads from burning cells to neighbors based on Huygens-correct
radial speeds.
"""
struct CAPropagation{N <: AbstractNeighborhood} <: AbstractPropagationModel
    neighborhood::N
end

CAPropagation(; neighborhood::AbstractNeighborhood=Moore()) = CAPropagation(neighborhood)

# ============================================================================ #
#                      Grid Factories
# ============================================================================ #

"""
    levelset_grid(nx, ny; dx=30.0, x0=0.0, y0=0.0)

Create a level set `Grid` with `nx × ny` cells. State is `Matrix{Float64}`
(φ values: positive=unburned, negative=burned, zero=front).
"""
function levelset_grid(nx::Integer, ny::Integer; dx=30.0, x0=0.0, y0=0.0)
    T = typeof(float(dx))
    x_lo, y_lo = T(x0), T(y0)
    x_hi = x_lo + nx * T(dx)
    y_hi = y_lo + ny * T(dx)
    extent = Extent(X=(x_lo, x_hi), Y=(y_lo, y_hi))
    state = ones(T, ny, nx)
    layers = (t_ignite = fill(T(Inf), ny, nx),)
    Grid(nothing, extent, T(dx), zero(T), state, layers)
end

"""
    ca_grid(nx, ny; dx=30.0, x0=0.0, y0=0.0)

Create a cellular automata `Grid` with `nx × ny` cells. State is `Matrix{CellState}`.
"""
function ca_grid(nx::Integer, ny::Integer; dx=30.0, x0=0.0, y0=0.0)
    T = typeof(float(dx))
    x_lo, y_lo = T(x0), T(y0)
    x_hi = x_lo + nx * T(dx)
    y_hi = y_lo + ny * T(dx)
    extent = Extent(X=(x_lo, x_hi), Y=(y_lo, y_hi))
    state = fill(UNBURNED, ny, nx)
    layers = (t_ignite = fill(T(Inf), ny, nx), t_arrival = fill(T(Inf), ny, nx))
    Grid(nothing, extent, T(dx), zero(T), state, layers)
end

# ============================================================================ #
#                     Grid Operations
# ============================================================================ #

#-----------------------------------------------------------------------------# ignite! (level set)
"""
    ignite!(g::Grid, cx, cy, r)

Set a circular ignition at center `(cx, cy)` with radius `r` (all in meters).
"""
function ignite!(g::Grid{C, T, <:AbstractFloat}, cx, cy, r) where {C, T}
    xs = xcoords(g)
    ys = ycoords(g)
    for j in eachindex(xs), i in eachindex(ys)
        d = hypot(xs[j] - cx, ys[i] - cy) - r
        if d < g.state[i, j]
            was_unburned = g.state[i, j] >= 0
            g.state[i, j] = d
            if was_unburned && d < 0 && isinf(g.layers.t_ignite[i, j])
                g.layers.t_ignite[i, j] = g.t
            end
        end
    end
    g
end

#-----------------------------------------------------------------------------# ignite! (CA)
function ignite!(g::Grid{C, T, CellState}, cx, cy, r) where {C, T}
    xs = xcoords(g)
    ys = ycoords(g)
    for j in eachindex(xs), i in eachindex(ys)
        if hypot(xs[j] - cx, ys[i] - cy) <= r && g.state[i, j] == UNBURNED
            g.state[i, j] = BURNING
            g.layers.t_ignite[i, j] = g.t
        end
    end
    g
end

#-----------------------------------------------------------------------------# set_unburnable! (level set)
"""
    set_unburnable!(g::Grid, cx, cy, r)

Mark all cells within radius `r` of `(cx, cy)` as unburnable.
"""
function set_unburnable!(g::Grid{C, T, <:AbstractFloat}, cx, cy, r) where {C, T}
    xs = xcoords(g)
    ys = ycoords(g)
    for j in eachindex(xs), i in eachindex(ys)
        if hypot(xs[j] - cx, ys[i] - cy) <= r
            g.layers.t_ignite[i, j] = eltype(g.layers.t_ignite)(NaN)
        end
    end
    g
end

#-----------------------------------------------------------------------------# set_unburnable! (CA)
function set_unburnable!(g::Grid{C, T, CellState}, cx, cy, r) where {C, T}
    xs = xcoords(g)
    ys = ycoords(g)
    for j in eachindex(xs), i in eachindex(ys)
        if hypot(xs[j] - cx, ys[i] - cy) <= r
            g.state[i, j] = UNBURNABLE
            g.layers.t_ignite[i, j] = eltype(g.layers.t_ignite)(NaN)
        end
    end
    g
end

#-----------------------------------------------------------------------------# Queries
burned(g::Grid{C, T, <:AbstractFloat}) where {C, T} = g.state .< 0
burned(g::Grid{C, T, CellState}) where {C, T} = map(==(BURNED), g.state)
burning(g::Grid{C, T, CellState}) where {C, T} = map(==(BURNING), g.state)
burnable(g::Grid{C, T, <:AbstractFloat}) where {C, T} = .!isnan.(g.layers.t_ignite)
burnable(g::Grid{C, T, CellState}) where {C, T} = map(!=(UNBURNABLE), g.state)

burn_area(g::Grid{C, T, <:AbstractFloat}) where {C, T} = count(<(0), g.state) * g.dx^2
function burn_area(g::Grid{C, T, CellState}) where {C, T}
    n = count(c -> c == BURNING || c == BURNED, g.state)
    n * g.dx^2
end

# ============================================================================ #
#                    Directional Speed
# ============================================================================ #

"""
    directional_speed(model::FireModel, t, x, y, nx, ny)

Compute the direction-dependent normal spread rate [m/min] for direction `(nx, ny)`.
"""
function directional_speed(fm::FireModel, t, x, y, nx, ny)
    hb = head_and_base(fm.ros, fm.env, fm.shape, t, x, y)
    cos_theta = nx * hb.push_x + ny * hb.push_y
    normal_speed(fm.shape, hb.R_head, hb.R_base, cos_theta, hb.LB)
end

# ============================================================================ #
#                    Spread Rate Field (Level Set)
# ============================================================================ #

"""
    spread_rate_field!(F, model::FireModel, grid::Grid)

Fill matrix `F` with direction-dependent spread rates at each cell center.
"""
function spread_rate_field!(F::AbstractMatrix, fm::FireModel, grid::Grid)
    xs = xcoords(grid)
    ys = ycoords(grid)
    t = grid.t
    φ = grid.state
    nrows, ncols = size(φ)
    dx = grid.dx
    t_ignite = grid.layers.t_ignite

    for j in eachindex(xs), i in eachindex(ys)
        if isnan(t_ignite[i, j])
            F[i, j] = zero(eltype(F))
            continue
        end

        dφdx = if j == 1
            (φ[i, 2] - φ[i, 1]) / dx
        elseif j == ncols
            (φ[i, ncols] - φ[i, ncols-1]) / dx
        else
            (φ[i, j+1] - φ[i, j-1]) / (2dx)
        end
        dφdy = if i == 1
            (φ[2, j] - φ[1, j]) / dx
        elseif i == nrows
            (φ[nrows, j] - φ[nrows-1, j]) / dx
        else
            (φ[i+1, j] - φ[i-1, j]) / (2dx)
        end
        grad = hypot(dφdx, dφdy)

        hb = head_and_base(fm.ros, fm.env, fm.shape, t, xs[j], ys[i])
        if grad > 0
            cos_theta = (dφdx * hb.push_x + dφdy * hb.push_y) / grad
            F[i, j] = normal_speed(fm.shape, hb.R_head, hb.R_base, cos_theta, hb.LB)
        else
            F[i, j] = hb.R_head
        end
    end
    F
end

# ============================================================================ #
#              Level Set Helpers: _phi_safe, Curvature
# ============================================================================ #

@inline function _phi_safe(φ, i, j, ny, nx, ::ZeroNeumann)
    @inbounds φ[clamp(i, 1, ny), clamp(j, 1, nx)]
end

@inline function _phi_safe(φ, i, j, ny, nx, ::Dirichlet)
    @inbounds φ[clamp(i, 1, ny), clamp(j, 1, nx)]
end

@inline function _phi_safe(φ, i, j, ny, nx, ::Periodic)
    @inbounds φ[mod1(i, ny), mod1(j, nx)]
end

@inline function _curvature(φ, i, j, ny, nx, dx, bc)
    φ_c  = _phi_safe(φ, i, j,     ny, nx, bc)
    φ_xp = _phi_safe(φ, i, j + 1, ny, nx, bc)
    φ_xm = _phi_safe(φ, i, j - 1, ny, nx, bc)
    φ_yp = _phi_safe(φ, i + 1, j, ny, nx, bc)
    φ_ym = _phi_safe(φ, i - 1, j, ny, nx, bc)

    φ_x = (φ_xp - φ_xm) / (2 * dx)
    φ_y = (φ_yp - φ_ym) / (2 * dx)
    φ_xx = (φ_xp - 2φ_c + φ_xm) / dx^2
    φ_yy = (φ_yp - 2φ_c + φ_ym) / dx^2

    φ_xpyp = _phi_safe(φ, i + 1, j + 1, ny, nx, bc)
    φ_xpym = _phi_safe(φ, i - 1, j + 1, ny, nx, bc)
    φ_xmyp = _phi_safe(φ, i + 1, j - 1, ny, nx, bc)
    φ_xmym = _phi_safe(φ, i - 1, j - 1, ny, nx, bc)
    φ_xy = (φ_xpyp - φ_xpym - φ_xmyp + φ_xmym) / (4 * dx^2)

    grad_sq = φ_x^2 + φ_y^2
    grad_mag = sqrt(grad_sq)
    grad_mag < eps(typeof(grad_mag)) && return zero(typeof(grad_mag))
    return (φ_xx * φ_y^2 - 2 * φ_x * φ_y * φ_xy + φ_yy * φ_x^2) / (grad_sq * grad_mag)
end

# ============================================================================ #
#              Level Set Helpers: Superbee
# ============================================================================ #

@inline function _half_superbee(r)
    max(zero(r), max(min(r / 2, one(r)), min(r, one(r) / 2)))
end

@inline function _normal_xy(φ, i, j, ny, nx, dx, bc)
    φxp = _phi_safe(φ, i, j + 1, ny, nx, bc)
    φxm = _phi_safe(φ, i, j - 1, ny, nx, bc)
    φyp = _phi_safe(φ, i + 1, j, ny, nx, bc)
    φym = _phi_safe(φ, i - 1, j, ny, nx, bc)
    dφdx = (φxp - φxm) / (2 * dx)
    dφdy = (φyp - φym) / (2 * dx)
    grad = hypot(dφdx, dφdy)
    ε = eps(typeof(grad))
    safe_grad = max(grad, ε)
    return dφdx / safe_grad, dφdy / safe_grad
end

@inline function _superbee_gradients(φ, ux, uy, i, j, ny, nx, dx, bc, grad_clamp)
    if ux >= 0
        φ_c  = _phi_safe(φ, i, j, ny, nx, bc)
        φ_m  = _phi_safe(φ, i, j - 1, ny, nx, bc)
        φ_mm = _phi_safe(φ, i, j - 2, ny, nx, bc)
        d_main = (φ_c - φ_m) / dx
        d_far  = (φ_m - φ_mm) / dx
        r = abs(d_main) > eps(typeof(d_main)) ? d_far / d_main : zero(d_main)
        ψ = _half_superbee(r)
        dφdx = d_main + ψ * (d_main - d_far)
    else
        φ_c  = _phi_safe(φ, i, j, ny, nx, bc)
        φ_p  = _phi_safe(φ, i, j + 1, ny, nx, bc)
        φ_pp = _phi_safe(φ, i, j + 2, ny, nx, bc)
        d_main = (φ_p - φ_c) / dx
        d_far  = (φ_pp - φ_p) / dx
        r = abs(d_main) > eps(typeof(d_main)) ? d_far / d_main : zero(d_main)
        ψ = _half_superbee(r)
        dφdx = d_main + ψ * (d_main - d_far)
    end

    if uy >= 0
        φ_c  = _phi_safe(φ, i, j, ny, nx, bc)
        φ_m  = _phi_safe(φ, i - 1, j, ny, nx, bc)
        φ_mm = _phi_safe(φ, i - 2, j, ny, nx, bc)
        d_main = (φ_c - φ_m) / dx
        d_far  = (φ_m - φ_mm) / dx
        r = abs(d_main) > eps(typeof(d_main)) ? d_far / d_main : zero(d_main)
        ψ = _half_superbee(r)
        dφdy = d_main + ψ * (d_main - d_far)
    else
        φ_c  = _phi_safe(φ, i, j, ny, nx, bc)
        φ_p  = _phi_safe(φ, i + 1, j, ny, nx, bc)
        φ_pp = _phi_safe(φ, i + 2, j, ny, nx, bc)
        d_main = (φ_p - φ_c) / dx
        d_far  = (φ_pp - φ_p) / dx
        r = abs(d_main) > eps(typeof(d_main)) ? d_far / d_main : zero(d_main)
        ψ = _half_superbee(r)
        dφdy = d_main + ψ * (d_main - d_far)
    end

    dφdx = clamp(dφdx, -grad_clamp, grad_clamp)
    dφdy = clamp(dφdy, -grad_clamp, grad_clamp)
    return dφdx, dφdy
end

# ============================================================================ #
#              Level Set Helpers: WENO5
# ============================================================================ #

@inline function _weno5_core(v1, v2, v3, v4, v5)
    s1 = v1 / 3 - 7v2 / 6 + 11v3 / 6
    s2 = -v2 / 6 + 5v3 / 6 + v4 / 3
    s3 = v3 / 3 + 5v4 / 6 - v5 / 6

    β1 = (13 / 12) * (v1 - 2v2 + v3)^2 + (1 / 4) * (v1 - 4v2 + 3v3)^2
    β2 = (13 / 12) * (v2 - 2v3 + v4)^2 + (1 / 4) * (v2 - v4)^2
    β3 = (13 / 12) * (v3 - 2v4 + v5)^2 + (1 / 4) * (3v3 - 4v4 + v5)^2

    ε = 1e-6
    α1 = oftype(v1, 0.1) / (ε + β1)^2
    α2 = oftype(v1, 0.6) / (ε + β2)^2
    α3 = oftype(v1, 0.3) / (ε + β3)^2
    sum_α = α1 + α2 + α3

    return (α1 * s1 + α2 * s2 + α3 * s3) / sum_α
end

@inline function _weno5_minus_x(φ, i, j, ny, nx, dx, bc)
    v1 = _phi_safe(φ, i, j - 2, ny, nx, bc) - _phi_safe(φ, i, j - 3, ny, nx, bc)
    v2 = _phi_safe(φ, i, j - 1, ny, nx, bc) - _phi_safe(φ, i, j - 2, ny, nx, bc)
    v3 = _phi_safe(φ, i, j,     ny, nx, bc) - _phi_safe(φ, i, j - 1, ny, nx, bc)
    v4 = _phi_safe(φ, i, j + 1, ny, nx, bc) - _phi_safe(φ, i, j,     ny, nx, bc)
    v5 = _phi_safe(φ, i, j + 2, ny, nx, bc) - _phi_safe(φ, i, j + 1, ny, nx, bc)
    _weno5_core(v1, v2, v3, v4, v5) / dx
end

@inline function _weno5_plus_x(φ, i, j, ny, nx, dx, bc)
    v1 = _phi_safe(φ, i, j + 3, ny, nx, bc) - _phi_safe(φ, i, j + 2, ny, nx, bc)
    v2 = _phi_safe(φ, i, j + 2, ny, nx, bc) - _phi_safe(φ, i, j + 1, ny, nx, bc)
    v3 = _phi_safe(φ, i, j + 1, ny, nx, bc) - _phi_safe(φ, i, j,     ny, nx, bc)
    v4 = _phi_safe(φ, i, j,     ny, nx, bc) - _phi_safe(φ, i, j - 1, ny, nx, bc)
    v5 = _phi_safe(φ, i, j - 1, ny, nx, bc) - _phi_safe(φ, i, j - 2, ny, nx, bc)
    _weno5_core(v1, v2, v3, v4, v5) / dx
end

@inline function _weno5_minus_y(φ, i, j, ny, nx, dx, bc)
    v1 = _phi_safe(φ, i - 2, j, ny, nx, bc) - _phi_safe(φ, i - 3, j, ny, nx, bc)
    v2 = _phi_safe(φ, i - 1, j, ny, nx, bc) - _phi_safe(φ, i - 2, j, ny, nx, bc)
    v3 = _phi_safe(φ, i,     j, ny, nx, bc) - _phi_safe(φ, i - 1, j, ny, nx, bc)
    v4 = _phi_safe(φ, i + 1, j, ny, nx, bc) - _phi_safe(φ, i,     j, ny, nx, bc)
    v5 = _phi_safe(φ, i + 2, j, ny, nx, bc) - _phi_safe(φ, i + 1, j, ny, nx, bc)
    _weno5_core(v1, v2, v3, v4, v5) / dx
end

@inline function _weno5_plus_y(φ, i, j, ny, nx, dx, bc)
    v1 = _phi_safe(φ, i + 3, j, ny, nx, bc) - _phi_safe(φ, i + 2, j, ny, nx, bc)
    v2 = _phi_safe(φ, i + 2, j, ny, nx, bc) - _phi_safe(φ, i + 1, j, ny, nx, bc)
    v3 = _phi_safe(φ, i + 1, j, ny, nx, bc) - _phi_safe(φ, i,     j, ny, nx, bc)
    v4 = _phi_safe(φ, i,     j, ny, nx, bc) - _phi_safe(φ, i - 1, j, ny, nx, bc)
    v5 = _phi_safe(φ, i - 1, j, ny, nx, bc) - _phi_safe(φ, i - 2, j, ny, nx, bc)
    _weno5_core(v1, v2, v3, v4, v5) / dx
end

function _weno5_rhs!(φ_out, φ_in, F, dt, ny, nx, dx, bc, phi_clamp)
    T = eltype(φ_in)
    z = zero(T)
    Threads.@threads for j in 1:nx
        for i in 1:ny
            if _skip_update(i, j, ny, nx, bc)
                φ_out[i, j] = φ_in[i, j]
                continue
            end

            Fij = F[i, j]
            if Fij <= z
                φ_out[i, j] = φ_in[i, j]
                continue
            end

            dxm = _weno5_minus_x(φ_in, i, j, ny, nx, dx, bc)
            dxp = _weno5_plus_x(φ_in, i, j, ny, nx, dx, bc)
            dym = _weno5_minus_y(φ_in, i, j, ny, nx, dx, bc)
            dyp = _weno5_plus_y(φ_in, i, j, ny, nx, dx, bc)

            Dxm_plus = max(dxm, z)
            Dxp_minus = min(dxp, z)
            Dym_plus = max(dym, z)
            Dyp_minus = min(dyp, z)

            grad_sq = max(Dxm_plus, -Dxp_minus)^2 + max(Dym_plus, -Dyp_minus)^2
            if grad_sq <= z
                φ_out[i, j] = φ_in[i, j]
                continue
            end

            new_phi = φ_in[i, j] - dt * Fij * sqrt(grad_sq)
            new_phi = isnan(new_phi) ? one(T) : clamp(new_phi, -phi_clamp, phi_clamp)
            φ_out[i, j] = new_phi
        end
    end
end

# ============================================================================ #
#                Level Set: advance!
# ============================================================================ #

function advance!(g::Grid{C, GT, <:AbstractFloat}, F::AbstractMatrix, dt, prop::LevelSetPropagation) where {C, GT}
    _advance_ls!(g, F, dt, prop.solver, prop.bc)
end

#-----------------------------------------------------------------------------# Godunov
function _advance_ls!(g, F, dt, ::Godunov, bc)
    φ = g.state
    ny, nx = size(φ)
    dx = g.dx
    T = eltype(φ)
    z = zero(T)

    φ_old = copy(φ)
    t_ignite = g.layers.t_ignite
    t_now = g.t + dt

    Threads.@threads for j in 1:nx
        for i in 1:ny
            _skip_update(i, j, ny, nx, bc) && continue
            Fij = F[i, j]
            Fij > z || continue

            dxm = _Dxm(φ_old, i, j, dx, bc)
            dxp = _Dxp(φ_old, i, j, nx, dx, bc)
            dym = _Dym(φ_old, i, j, dx, bc)
            dyp = _Dyp(φ_old, i, j, ny, dx, bc)

            Dxm_plus = max(dxm, z)
            Dxp_minus = min(dxp, z)
            Dym_plus = max(dym, z)
            Dyp_minus = min(dyp, z)

            grad_sq = max(Dxm_plus, -Dxp_minus)^2 + max(Dym_plus, -Dyp_minus)^2
            grad_sq > z || continue
            grad_mag = sqrt(grad_sq)

            new_phi = φ_old[i, j] - dt * Fij * grad_mag
            if φ_old[i, j] >= z && new_phi < z && !isnan(t_ignite[i, j]) && isinf(t_ignite[i, j])
                t_ignite[i, j] = t_now
            end
            φ[i, j] = new_phi
        end
    end

    g.t = t_now
    g
end

#-----------------------------------------------------------------------------# Superbee
function _advance_ls!(g, F, dt, solver::Superbee, bc)
    φ = g.state
    ny, nx = size(φ)
    dx = g.dx
    T = eltype(φ)
    z = zero(T)
    phi_clamp = T(solver.phi_clamp)
    grad_clamp = T(solver.grad_clamp)

    φ_old = copy(φ)
    t_ignite = g.layers.t_ignite
    t_now = g.t + dt

    # Stage 1: Forward Euler
    φ_buf = copy(φ)
    Threads.@threads for j in 1:nx
        for i in 1:ny
            _skip_update(i, j, ny, nx, bc) && continue
            Fij = F[i, j]
            Fij > z || continue

            nx_n, ny_n = _normal_xy(φ_buf, i, j, ny, nx, dx, bc)
            ux = Fij * nx_n
            uy = Fij * ny_n
            dφdx, dφdy = _superbee_gradients(φ_buf, ux, uy, i, j, ny, nx, dx, bc, grad_clamp)

            new_phi = φ_buf[i, j] - dt * (ux * dφdx + uy * dφdy)
            new_phi = isnan(new_phi) ? one(T) : clamp(new_phi, -phi_clamp, phi_clamp)
            φ[i, j] = new_phi
        end
    end

    # Stage 2: Recompute from stage-1 values, then average with φ_old
    φ_buf .= φ
    Threads.@threads for j in 1:nx
        for i in 1:ny
            _skip_update(i, j, ny, nx, bc) && continue
            Fij = F[i, j]
            Fij > z || continue

            nx_n, ny_n = _normal_xy(φ_buf, i, j, ny, nx, dx, bc)
            ux = Fij * nx_n
            uy = Fij * ny_n
            dφdx, dφdy = _superbee_gradients(φ_buf, ux, uy, i, j, ny, nx, dx, bc, grad_clamp)

            φ_star = φ_buf[i, j] - dt * (ux * dφdx + uy * dφdy)
            new_phi = T(0.5) * (φ_old[i, j] + φ_star)
            new_phi = isnan(new_phi) ? one(T) : clamp(new_phi, -phi_clamp, phi_clamp)
            φ[i, j] = new_phi
        end
    end

    # Record ignition times
    for j in 1:nx, i in 1:ny
        if φ_old[i, j] >= z && φ[i, j] < z && !isnan(t_ignite[i, j]) && isinf(t_ignite[i, j])
            t_ignite[i, j] = t_now
        end
    end

    g.t = t_now
    g
end

#-----------------------------------------------------------------------------# WENO5
function _advance_ls!(g, F, dt, solver::WENO5, bc)
    φ = g.state
    ny, nx = size(φ)
    dx = g.dx
    T = eltype(φ)
    z = zero(T)
    phi_clamp = T(solver.phi_clamp)

    φ_n = copy(φ)
    φ_1 = similar(φ)
    φ_tmp = similar(φ)

    # SSP-RK3
    _weno5_rhs!(φ_1, φ_n, F, dt, ny, nx, dx, bc, phi_clamp)
    _weno5_rhs!(φ_tmp, φ_1, F, dt, ny, nx, dx, bc, phi_clamp)
    φ_2 = @. T(3 / 4) * φ_n + T(1 / 4) * φ_tmp
    _weno5_rhs!(φ_tmp, φ_2, F, dt, ny, nx, dx, bc, phi_clamp)
    @. φ = T(1 / 3) * φ_n + T(2 / 3) * φ_tmp

    # Record ignition times
    t_ignite = g.layers.t_ignite
    t_now = g.t + dt
    for j in 1:nx, i in 1:ny
        if φ_n[i, j] >= z && φ[i, j] < z && !isnan(t_ignite[i, j]) && isinf(t_ignite[i, j])
            t_ignite[i, j] = t_now
        end
    end

    g.t = t_now
    g
end

# ============================================================================ #
#                Level Set: CFL + Reinitialize
# ============================================================================ #

"""
    cfl_dt(g::Grid, F; cfl=0.5, curvature=0.0)

Compute a CFL-stable time step for the level set equation given spread rate field `F`.
"""
function cfl_dt(g::Grid{C, GT, <:AbstractFloat}, F::AbstractMatrix; cfl=0.5, curvature=0.0) where {C, GT}
    T = eltype(g.state)
    Fmax = maximum(F)
    Fmax > 0 || return T(Inf)
    dt_adv = T(cfl) * g.dx / Fmax
    if curvature > 0
        dt_curv = g.dx^2 / (2 * abs(curvature))
        return min(dt_adv, dt_curv)
    end
    dt_adv
end

_smoothed_sign(φ, h) = φ / hypot(φ, h)

"""
    reinitialize!(g::Grid, method::IterativeReinit, bc::AbstractBoundaryCondition)

Reinitialize φ toward a signed distance function.
"""
function reinitialize!(g::Grid{C, GT, <:AbstractFloat}, method::IterativeReinit, bc::AbstractBoundaryCondition) where {C, GT}
    φ = g.state
    ny, nx = size(φ)
    dx = g.dx
    T = eltype(φ)
    h = dx
    dτ = h / 2
    z = zero(T)

    was_nonneg = φ .>= z

    for _ in 1:method.iterations
        φ_old = copy(φ)
        Threads.@threads for j in 1:nx
            for i in 1:ny
                _skip_update(i, j, ny, nx, bc) && continue
                φ_ij = φ_old[i, j]
                S = _smoothed_sign(φ_ij, h)

                dxm = _Dxm(φ_old, i, j, dx, bc)
                dxp = _Dxp(φ_old, i, j, nx, dx, bc)
                dym = _Dym(φ_old, i, j, dx, bc)
                dyp = _Dyp(φ_old, i, j, ny, dx, bc)

                if S > z
                    a = max(max(dxm, z), -min(dxp, z))
                    b = max(max(dym, z), -min(dyp, z))
                else
                    a = max(-min(dxm, z), max(dxp, z))
                    b = max(-min(dym, z), max(dyp, z))
                end

                grad_mag = hypot(a, b)
                φ[i, j] = φ_ij - dτ * S * (grad_mag - one(T))
            end
        end
    end

    _update_ignition_after_reinit!(g, was_nonneg)
    g
end

function _update_ignition_after_reinit!(g, was_nonneg)
    φ = g.state
    t_ignite = g.layers.t_ignite
    t_now = g.t
    z = zero(eltype(φ))
    for j in axes(φ, 2), i in axes(φ, 1)
        if was_nonneg[i, j] && φ[i, j] < z && !isnan(t_ignite[i, j]) && isinf(t_ignite[i, j])
            t_ignite[i, j] = t_now
        end
    end
end

# ============================================================================ #
#              Burnout / Burnin / Curvature Scaling
# ============================================================================ #

_scale_burnout!(F, g, ::NoBurnout) = nothing

function _scale_burnout!(F, g, bo::AbstractBurnout)
    t_ignite = g.layers.t_ignite
    for j in axes(F, 2), i in axes(F, 1)
        t_ig = t_ignite[i, j]
        isfinite(t_ig) && (F[i, j] *= bo(g.t - t_ig))
    end
end

_scale_burnin!(F, g, ::NoBurnin) = nothing

function _scale_burnin!(F, g, bi::AbstractBurnin)
    t_ignite = g.layers.t_ignite
    for j in axes(F, 2), i in axes(F, 1)
        t_ig = t_ignite[i, j]
        isfinite(t_ig) && (F[i, j] *= bi(g.t - t_ig))
    end
end

function _apply_curvature!(F, g, b, bc)
    φ = g.state
    ny, nx = size(φ)
    dx = g.dx
    z = zero(eltype(F))
    for j in 1:nx, i in 1:ny
        κ = _curvature(φ, i, j, ny, nx, dx, bc)
        F[i, j] = max(F[i, j] - b * κ, z)
    end
end

# ============================================================================ #
#                   CA: CFL + advance!
# ============================================================================ #

function _ca_cfl_dt(g::Grid{C, GT, CellState}, fm::FireModel; cfl=0.5) where {C, GT}
    xs = xcoords(g)
    ys = ycoords(g)
    T = typeof(g.t)
    R_max = zero(T)
    for j in eachindex(xs), i in eachindex(ys)
        g.state[i, j] == BURNING || continue
        hb = head_and_base(fm.ros, fm.env, fm.shape, g.t, xs[j], ys[i])
        R_max = max(R_max, hb.R_head)
    end
    R_max > 0 || return T(Inf)
    return T(cfl) * g.dx / R_max
end

"""
    advance!(g::Grid{..., CellState}, model::FireModel, dt, prop::CAPropagation; burnout, burnin, residence_time)

Advance the CA fire simulation by one time step using a travel-time approach
with Huygens-correct radial speeds.
"""
function advance!(g::Grid{C, GT, CellState}, fm::FireModel, dt, prop::CAPropagation;
        burnout::AbstractBurnout=NoBurnout(),
        burnin::AbstractBurnin=NoBurnin(),
        residence_time=Inf) where {C, GT}
    state = g.state
    ny, nx = size(state)
    dx = g.dx
    offsets = _offsets(prop.neighborhood)
    xs = xcoords(g)
    ys = ycoords(g)
    t_ignite = g.layers.t_ignite
    t_arrival = g.layers.t_arrival
    T = typeof(g.t)
    t_now = g.t + dt

    # Phase 1: Compute arrival times from BURNING -> UNBURNED neighbors
    for j in 1:nx, i in 1:ny
        state[i, j] == BURNING || continue

        t_burn = g.t - t_ignite[i, j]
        scale = burnout(t_burn) * burnin(t_burn)
        scale > zero(T) || continue

        x_src = xs[j]
        y_src = ys[i]

        hb = head_and_base(fm.ros, fm.env, fm.shape, g.t, x_src, y_src)

        for (di, dj) in offsets
            ni, nj = i + di, j + dj
            (1 <= ni <= ny && 1 <= nj <= nx) || continue
            state[ni, nj] == UNBURNED || continue

            delta_x = T(dj) * dx
            delta_y = T(di) * dx
            dist = hypot(delta_x, delta_y)
            norm_x = delta_x / dist
            norm_y = delta_y / dist

            cos_theta = norm_x * hb.push_x + norm_y * hb.push_y
            R = radial_speed(fm.shape, hb.R_head, hb.R_base, cos_theta, hb.LB) * scale
            R > zero(T) || continue

            t_arr = t_ignite[i, j] + dist / R
            t_arrival[ni, nj] = min(t_arrival[ni, nj], t_arr)
        end
    end

    # Phase 2: Ignite cells whose arrival time <= t_now
    for j in 1:nx, i in 1:ny
        if state[i, j] == UNBURNED && t_arrival[i, j] <= t_now
            state[i, j] = BURNING
            t_ignite[i, j] = t_arrival[i, j]
            t_arrival[i, j] = T(Inf)
        end
    end

    # Phase 3: Burnout (BURNING -> BURNED)
    for j in 1:nx, i in 1:ny
        if state[i, j] == BURNING
            elapsed = t_now - t_ignite[i, j]
            if elapsed >= residence_time
                state[i, j] = BURNED
            end
        end
    end

    g.t = t_now
    g
end

# ============================================================================ #
#                          Trace
# ============================================================================ #

"""
    Trace(grid::Grid, every::Integer)

Records snapshots of grid state during [`simulate!`](@ref).
"""
struct Trace{T, M}
    stack::Vector{Tuple{T, M}}
    every::Int
end

function Trace(grid::Grid, every::Integer)
    T = typeof(grid.t)
    snapshot = copy(grid.state)
    M = typeof(snapshot)
    Trace{T, M}(Tuple{T, M}[(grid.t, snapshot)], every)
end

function _record!(trace::Trace, grid::Grid)
    push!(trace.stack, (grid.t, copy(grid.state)))
end

# ============================================================================ #
#                        simulate!
# ============================================================================ #

"""
    simulate!(grid, model, prop::LevelSetPropagation; steps=100, dt=nothing, cfl=0.5, burnout=nothing, burnin=nothing, trace=nothing, progress=false)

Run the level set simulation.
"""
function simulate!(grid::Grid{C, GT, <:AbstractFloat}, fm::FireModel,
        prop::LevelSetPropagation;
        steps::Int=100, dt=nothing, cfl=0.5,
        burnout=nothing, burnin=nothing,
        trace=nothing, progress::Bool=false) where {C, GT}
    bo = _coerce_burnout(burnout)
    bi = _coerce_burnin(burnin)
    F = similar(grid.state)
    for step in 1:steps
        spread_rate_field!(F, fm, grid)
        _scale_burnout!(F, grid, bo)
        _scale_burnin!(F, grid, bi)
        prop.curvature > 0 && _apply_curvature!(F, grid, prop.curvature, prop.bc)
        step_dt = dt === nothing ? cfl_dt(grid, F; cfl=cfl, curvature=prop.curvature) : dt
        advance!(grid, F, step_dt, prop)
        step % prop.reinit_every == 0 && reinitialize!(grid, prop.reinit, prop.bc)
        trace !== nothing && step % trace.every == 0 && _record!(trace, grid)
        progress && step % max(1, steps ÷ 100) == 0 && _print_ls_progress(step, steps, grid)
    end
    progress && println()
    grid
end

function _print_ls_progress(step, steps, grid)
    pct = round(Int, 100 * step / steps)
    n_burned = count(<(0), grid.state)
    n_total = length(grid.state)
    print("\r  step $step/$steps ($pct%) | t = $(round(grid.t, digits=2)) min | burned = $n_burned/$n_total")
end

"""
    simulate!(grid, model, prop::CAPropagation; steps=100, dt=nothing, cfl=0.5, burnout=nothing, burnin=nothing, residence_time=Inf, trace=nothing, progress=false)

Run the cellular automata simulation.
"""
function simulate!(grid::Grid{C, GT, CellState}, fm::FireModel,
        prop::CAPropagation;
        steps::Int=100, dt=nothing, cfl=0.5,
        burnout=nothing, burnin=nothing,
        residence_time=Inf,
        trace=nothing, progress::Bool=false) where {C, GT}
    bo = _coerce_burnout(burnout)
    bi = _coerce_burnin(burnin)
    for step in 1:steps
        step_dt = dt === nothing ? _ca_cfl_dt(grid, fm; cfl=cfl) : dt
        advance!(grid, fm, step_dt, prop; burnout=bo, burnin=bi, residence_time=residence_time)
        trace !== nothing && step % trace.every == 0 && _record!(trace, grid)
        progress && step % max(1, steps ÷ 100) == 0 && _print_ca_progress(step, steps, grid)
    end
    progress && println()
    grid
end

function _print_ca_progress(step, steps, grid)
    pct = round(Int, 100 * step / steps)
    n_burning = count(==(BURNING), grid.state)
    n_burned = count(c -> c == BURNING || c == BURNED, grid.state)
    n_total = length(grid.state)
    print("\r  step $step/$steps ($pct%) | t = $(round(grid.t, digits=2)) min | burned = $n_burned/$n_total | burning = $n_burning")
end

end # module
