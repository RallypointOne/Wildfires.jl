module LevelSet

export LevelSetGrid, xcoords, ycoords, burned, burn_area, ignite!, advance!, reinitialize!, cfl_dt, set_unburnable!, burnable
export AbstractBoundaryCondition, ZeroNeumann, Dirichlet, Periodic

#=
    Level Set Method for Fire Spread

    The fire front is the zero contour of a signed distance function φ(x, y, t):
      - φ < 0 : burned
      - φ = 0 : fire front
      - φ > 0 : unburned

    Evolution follows the Hamilton-Jacobi equation:
      ∂φ/∂t + F|∇φ| = 0
    where F(x,y,t) ≥ 0 is the local fire spread rate (e.g. from Rothermel).

    References:
    - Osher, S. & Sethian, J.A. (1988). Fronts propagating with curvature-dependent
      speed. J. Computational Physics, 79(1), 12–49.
    - Mallet, V., Keyes, D.E., & Fendell, F.E. (2009). Modeling wildland fire
      propagation with level set methods. Computers & Mathematics with Applications.
=#

using LinearAlgebra: norm

#-----------------------------------------------------------------------------# Boundary Conditions
"""
    AbstractBoundaryCondition

Supertype for level set boundary conditions.

Subtypes control how `advance!` and `reinitialize!` handle grid edges.
Implement `_Dxm`, `_Dxp`, `_Dym`, `_Dyp` and optionally `_skip_update`
to define a custom boundary condition.
"""
abstract type AbstractBoundaryCondition end

"""
    ZeroNeumann()

Zero-gradient (Neumann) boundary condition.  At grid edges, finite differences
that would reference out-of-bounds cells return zero.  This is the default.
"""
struct ZeroNeumann <: AbstractBoundaryCondition end

"""
    Dirichlet()

Fixed-value (Dirichlet) boundary condition.  Edge cells are not updated by
`advance!` or `reinitialize!`, preserving their initial values.
"""
struct Dirichlet <: AbstractBoundaryCondition end

"""
    Periodic()

Periodic (wrap-around) boundary condition.  At grid edges, finite differences
use values from the opposite edge.
"""
struct Periodic <: AbstractBoundaryCondition end

#-----------------------------------------------------------------------------# Boundary-aware finite differences
# ZeroNeumann: missing neighbor → zero difference
@inline _Dxm(φ, i, j, dx, ::ZeroNeumann) = j > 1 ? (φ[i, j] - φ[i, j-1]) / dx : zero(eltype(φ))
@inline _Dxp(φ, i, j, nx, dx, ::ZeroNeumann) = j < nx ? (φ[i, j+1] - φ[i, j]) / dx : zero(eltype(φ))
@inline _Dym(φ, i, j, dy, ::ZeroNeumann) = i > 1 ? (φ[i, j] - φ[i-1, j]) / dy : zero(eltype(φ))
@inline _Dyp(φ, i, j, ny, dy, ::ZeroNeumann) = i < ny ? (φ[i+1, j] - φ[i, j]) / dy : zero(eltype(φ))

# Dirichlet: same differences as ZeroNeumann (edge cells are skipped via _skip_update)
@inline _Dxm(φ, i, j, dx, ::Dirichlet) = _Dxm(φ, i, j, dx, ZeroNeumann())
@inline _Dxp(φ, i, j, nx, dx, ::Dirichlet) = _Dxp(φ, i, j, nx, dx, ZeroNeumann())
@inline _Dym(φ, i, j, dy, ::Dirichlet) = _Dym(φ, i, j, dy, ZeroNeumann())
@inline _Dyp(φ, i, j, ny, dy, ::Dirichlet) = _Dyp(φ, i, j, ny, dy, ZeroNeumann())

# Periodic: wrap-around indices
@inline function _Dxm(φ, i, j, dx, ::Periodic)
    jm = j > 1 ? j - 1 : size(φ, 2)
    (φ[i, j] - φ[i, jm]) / dx
end
@inline function _Dxp(φ, i, j, nx, dx, ::Periodic)
    jp = j < nx ? j + 1 : 1
    (φ[i, jp] - φ[i, j]) / dx
end
@inline function _Dym(φ, i, j, dy, ::Periodic)
    im = i > 1 ? i - 1 : size(φ, 1)
    (φ[i, j] - φ[im, j]) / dy
end
@inline function _Dyp(φ, i, j, ny, dy, ::Periodic)
    ip = i < ny ? i + 1 : 1
    (φ[ip, j] - φ[i, j]) / dy
end

"""
    _skip_update(i, j, ny, nx, bc::AbstractBoundaryCondition) -> Bool

Return `true` if cell `(i, j)` should be skipped during the update.
"""
@inline _skip_update(i, j, ny, nx, ::AbstractBoundaryCondition) = false
@inline _skip_update(i, j, ny, nx, ::Dirichlet) = i == 1 || i == ny || j == 1 || j == nx

#-----------------------------------------------------------------------------# LevelSetGrid
"""
    LevelSetGrid{T, M, BC} <: AbstractMatrix{T}

A 2D grid representing a fire spread simulation via the level set method.

The grid stores a signed distance function `φ` where:
- `φ < 0` → burned
- `φ = 0` → fire front
- `φ > 0` → unburned

Ignition time is tracked per cell via `t_ignite`:
- `Inf`    → burnable, not yet ignited
- `NaN`    → unburnable (water, road, fuel break)
- finite   → ignited at that simulation time

# Fields
- `φ::M`          - Level set function values (`M <: AbstractMatrix{T}`)
- `t_ignite::M`   - Per-cell ignition time [min]
- `dx::T`         - Grid spacing in x [m]
- `dy::T`         - Grid spacing in y [m]
- `x0::T`         - x-coordinate of grid origin (lower-left) [m]
- `y0::T`         - y-coordinate of grid origin (lower-left) [m]
- `t::T`          - Current simulation time [min]
- `bc::BC`        - Boundary condition (`BC <: AbstractBoundaryCondition`)

# Constructor
    LevelSetGrid(nx, ny; dx=30.0, dy=dx, x0=0.0, y0=0.0, bc=ZeroNeumann())

Create an unburned grid with `nx × ny` cells (all burnable by default).

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)

grid = LevelSetGrid(100, 100, dx=30.0, bc=Periodic())
```
"""
mutable struct LevelSetGrid{T, M <: AbstractMatrix{T}, BC <: AbstractBoundaryCondition} <: AbstractMatrix{T}
    φ::M
    t_ignite::M
    dx::T
    dy::T
    x0::T
    y0::T
    t::T
    bc::BC
end

function LevelSetGrid(nx::Integer, ny::Integer; dx=30.0, dy=dx, x0=0.0, y0=0.0, bc::AbstractBoundaryCondition=ZeroNeumann())
    T = promote_type(typeof(dx), typeof(dy), typeof(x0), typeof(y0))
    φ = ones(T, ny, nx)  # all positive → unburned
    t_ignite = fill(T(Inf), ny, nx)
    LevelSetGrid(φ, t_ignite, T(dx), T(dy), T(x0), T(y0), zero(T), bc)
end

#-----------------------------------------------------------------------------# AbstractMatrix interface
Base.size(g::LevelSetGrid) = size(g.φ)
Base.getindex(g::LevelSetGrid, i...) = getindex(g.φ, i...)
Base.setindex!(g::LevelSetGrid, v, i...) = setindex!(g.φ, v, i...)

function Base.show(io::IO, ::MIME"text/plain", g::LevelSetGrid{T}) where {T}
    ny, nx = size(g.φ)
    nb = count(<(0), g.φ)
    bc_str = g.bc isa ZeroNeumann ? "" : ", bc=$(nameof(typeof(g.bc)))"
    n_unburnable = count(isnan, g.t_ignite)
    n_ignited = count(isfinite, g.t_ignite)
    ub_str = n_unburnable > 0 ? ", unburnable=$n_unburnable" : ""
    ig_str = n_ignited > 0 ? ", ignited=$n_ignited" : ""
    print(io, "LevelSetGrid{$T} $(nx)×$(ny) (t=$(g.t), burned=$(nb)/$(nx*ny)$(ub_str)$(ig_str)$(bc_str))")
end

function Base.show(io::IO, g::LevelSetGrid{T}) where {T}
    ny, nx = size(g.φ)
    print(io, "LevelSetGrid{$T}($(nx)×$(ny))")
end

#-----------------------------------------------------------------------------# Coordinates
"""
    xcoords(grid::LevelSetGrid)

Return the x-coordinates of grid cell centers.
"""
xcoords(g::LevelSetGrid) = range(g.x0 + g.dx / 2, step=g.dx, length=size(g.φ, 2))

"""
    ycoords(grid::LevelSetGrid)

Return the y-coordinates of grid cell centers.
"""
ycoords(g::LevelSetGrid) = range(g.y0 + g.dy / 2, step=g.dy, length=size(g.φ, 1))

#-----------------------------------------------------------------------------# Queries
"""
    burned(grid::LevelSetGrid)

Return a `BitMatrix` where `true` indicates burned cells (`φ < 0`).
"""
burned(g::LevelSetGrid) = g.φ .< 0

"""
    burn_area(grid::LevelSetGrid)

Return the total burned area [m²].
"""
burn_area(g::LevelSetGrid) = count(<(0), g.φ) * g.dx * g.dy

"""
    burnable(grid::LevelSetGrid)

Return a `BitMatrix` where `true` indicates burnable cells (i.e. `t_ignite` is not `NaN`).
"""
burnable(g::LevelSetGrid) = .!isnan.(g.t_ignite)

#-----------------------------------------------------------------------------# ignite!
"""
    ignite!(grid::LevelSetGrid, cx, cy, r)

Set a circular ignition at center `(cx, cy)` with radius `r` (all in meters).

Updates `φ` so that cells within radius `r` of `(cx, cy)` have `φ < 0` (burned)
while maintaining signed-distance-like values.

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)
```
"""
function ignite!(g::LevelSetGrid, cx, cy, r)
    xs = xcoords(g)
    ys = ycoords(g)
    for j in eachindex(xs), i in eachindex(ys)
        d = hypot(xs[j] - cx, ys[i] - cy) - r
        if d < g.φ[i, j]
            was_unburned = g.φ[i, j] >= 0
            g.φ[i, j] = d
            if was_unburned && d < 0 && isinf(g.t_ignite[i, j])
                g.t_ignite[i, j] = g.t
            end
        end
    end
    g
end

#-----------------------------------------------------------------------------# set_unburnable!
"""
    set_unburnable!(grid::LevelSetGrid, cx, cy, r)

Mark all cells within radius `r` of `(cx, cy)` as unburnable.

Unburnable cells always have a spread rate of zero, preventing fire from
entering them.  Useful for representing water bodies, roads, fuel breaks,
and pre-existing fire scars.

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
set_unburnable!(grid, 1500.0, 1500.0, 200.0)  # circular fuel break
```
"""
function set_unburnable!(g::LevelSetGrid, cx, cy, r)
    xs = xcoords(g)
    ys = ycoords(g)
    for j in eachindex(xs), i in eachindex(ys)
        if hypot(xs[j] - cx, ys[i] - cy) <= r
            g.t_ignite[i, j] = eltype(g.t_ignite)(NaN)
        end
    end
    g
end

#-----------------------------------------------------------------------------# advance!
"""
    advance!(grid::LevelSetGrid, F::AbstractMatrix, dt)

Advance the fire front by one time step `dt` [min] given a spread rate field
`F` [m/min] (same dimensions as `grid`).

Uses first-order upwind differencing for the Hamilton-Jacobi equation:

    φⁿ⁺¹ = φⁿ - dt · F · |∇φ|

where `|∇φ|` is computed with Godunov's upwind scheme.
"""
function advance!(g::LevelSetGrid{T}, F::AbstractMatrix, dt) where {T}
    φ = g.φ
    ny, nx = size(φ)
    dx, dy = g.dx, g.dy
    bc = g.bc
    z = zero(T)

    φ_old = copy(φ)
    t_ignite = g.t_ignite
    t_now = g.t + dt

    Threads.@threads for j in 1:nx
        for i in 1:ny
            _skip_update(i, j, ny, nx, bc) && continue

            Fij = F[i, j]
            Fij > z || continue

            # Upwind finite differences (Godunov)
            dxm = _Dxm(φ_old, i, j, dx, bc)
            dxp = _Dxp(φ_old, i, j, nx, dx, bc)
            dym = _Dym(φ_old, i, j, dy, bc)
            dyp = _Dyp(φ_old, i, j, ny, dy, bc)

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

#-----------------------------------------------------------------------------# cfl_dt
"""
    cfl_dt(grid::LevelSetGrid, F::AbstractMatrix; cfl=0.5)

Compute a CFL-stable time step for the level set equation given spread rate field `F`.

Returns `cfl * min(dx, dy) / max(F)`.  If `max(F) ≤ 0` (no active spread), returns `Inf`.

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
F = fill(10.0, size(grid))
cfl_dt(grid, F)  # 1.5
```
"""
function cfl_dt(g::LevelSetGrid{T}, F::AbstractMatrix; cfl=0.5) where {T}
    Fmax = maximum(F)
    Fmax > 0 || return T(Inf)
    T(cfl) * min(g.dx, g.dy) / Fmax
end

#-----------------------------------------------------------------------------# reinitialize!
"""
    reinitialize!(grid::LevelSetGrid; iterations=5)

Reinitialize `φ` toward a signed distance function using the iterative
fast-sweeping approach. This prevents `φ` from becoming too flat or too steep,
which degrades gradient accuracy.

Should be called periodically (e.g. every 5–10 time steps).
"""
function reinitialize!(g::LevelSetGrid{T}; iterations::Int=5) where {T}
    φ = g.φ
    ny, nx = size(φ)
    dx, dy = g.dx, g.dy
    bc = g.bc
    dτ = min(dx, dy) / 2  # pseudo-timestep
    z = zero(T)

    for _ in 1:iterations
        φ_old = copy(φ)
        Threads.@threads for j in 1:nx
            for i in 1:ny
                _skip_update(i, j, ny, nx, bc) && continue

                S = sign(φ_old[i, j])

                dxm = _Dxm(φ_old, i, j, dx, bc)
                dxp = _Dxp(φ_old, i, j, nx, dx, bc)
                dym = _Dym(φ_old, i, j, dy, bc)
                dyp = _Dyp(φ_old, i, j, ny, dy, bc)

                if S > 0
                    a = max(max(dxm, z), -min(dxp, z))
                    b = max(max(dym, z), -min(dyp, z))
                else
                    a = max(-min(dxm, z), max(dxp, z))
                    b = max(-min(dym, z), max(dyp, z))
                end

                grad_mag = hypot(a, b)
                φ[i, j] = φ_old[i, j] - dτ * S * (grad_mag - one(T))
            end
        end
    end
    g
end

end # module
