module LevelSet

export LevelSetGrid, xcoords, ycoords, burned, burn_area, ignite!, advance!, reinitialize!, cfl_dt, set_unburnable!, burnable
export AbstractBoundaryCondition, ZeroNeumann, Dirichlet, Periodic
export AbstractSolver, Godunov, Superbee, WENO5
export AbstractReinitMethod, IterativeReinit, NewtonReinit

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
using Interpolations: Interpolations, interpolate, scale, extrapolate, BSpline, Cubic, Line, OnGrid, Flat
using NearestNeighbors: KDTree, nn
using StaticArrays: SVector

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

#-----------------------------------------------------------------------------# Solvers
"""
    AbstractSolver

Supertype for level set advection solvers.

Subtypes control the numerical scheme used by [`advance!`](@ref) to evolve
the level set equation `φ_t + F|∇φ| = 0`.
"""
abstract type AbstractSolver end

"""
    Godunov()

First-order Godunov upwind scheme with forward Euler time stepping.
This is the default solver and reproduces the original `advance!` behavior.
"""
struct Godunov <: AbstractSolver end

"""
    Superbee(; phi_clamp=100.0, grad_clamp=1000.0)

Second-order Superbee flux limiter with RK2 (Heun's method) time stepping.

Converts the Hamilton-Jacobi equation to advection form using the front normal,
then applies the Superbee TVD limiter for sharper fire fronts with less numerical
diffusion. Based on the ELMFIRE approach.

# Fields
- `phi_clamp::T`  — clamp `φ` to `[-phi_clamp, phi_clamp]` after each stage
- `grad_clamp::T` — clamp flux-limited gradients to `[-grad_clamp, grad_clamp]`
"""
struct Superbee{T} <: AbstractSolver
    phi_clamp::T
    grad_clamp::T
end
Superbee(; phi_clamp=100.0, grad_clamp=1000.0) = Superbee(phi_clamp, grad_clamp)

"""
    WENO5(; phi_clamp=100.0)

Fifth-order WENO (Weighted Essentially Non-Oscillatory) scheme with SSP-RK3
(Strong Stability Preserving Runge-Kutta) time stepping.

Uses the Jiang-Shu WENO5 reconstruction for spatial derivatives with the
Godunov numerical Hamiltonian, providing 5th-order accuracy in smooth regions
while maintaining sharp resolution near discontinuities.

The stencil accesses `φ[i±3, j±3]` (3 ghost cells), wider than Godunov (1) or
Superbee (2).

# Fields
- `phi_clamp::T` — clamp `φ` to `[-phi_clamp, phi_clamp]` after each stage
"""
struct WENO5{T} <: AbstractSolver
    phi_clamp::T
end
WENO5(; phi_clamp=100.0) = WENO5(phi_clamp)

#-----------------------------------------------------------------------------# Reinitialization Methods
"""
    AbstractReinitMethod

Supertype for reinitialization methods used by [`reinitialize!`](@ref).
"""
abstract type AbstractReinitMethod end

"""
    IterativeReinit(; iterations=5)

PDE-based iterative reinitialization (Sussman et al. 1994).
This is the default method and reproduces the original `reinitialize!` behavior.
"""
struct IterativeReinit <: AbstractReinitMethod
    iterations::Int
end
IterativeReinit(; iterations=5) = IterativeReinit(iterations)

"""
    NewtonReinit(; upsample=8, maxiters=20, tol=1e-8)

Newton closest-point reinitialization for machine-precision signed distance
restoration. Uses cubic interpolation to locate the zero contour, then a
KD-tree for efficient nearest-neighbor queries.

# Fields
- `upsample::Int`   — subdivisions per interface cell for zero-contour sampling
- `maxiters::Int`    — maximum Newton iterations for projecting onto φ=0
- `tol::Float64`     — convergence tolerance for Newton projection
"""
struct NewtonReinit <: AbstractReinitMethod
    upsample::Int
    maxiters::Int
    tol::Float64
end
NewtonReinit(; upsample=8, maxiters=20, tol=1e-8) = NewtonReinit(upsample, maxiters, tol)

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
    advance!(grid::LevelSetGrid, F::AbstractMatrix, dt, [solver])

Advance the fire front by one time step `dt` [min] given a spread rate field
`F` [m/min] (same dimensions as `grid`).

The numerical scheme is selected by `solver`:
- `Godunov()` (default) — first-order upwind + forward Euler
- `Superbee()` — second-order Superbee flux limiter + RK2 (Heun's method)
"""
advance!(g::LevelSetGrid, F::AbstractMatrix, dt) = advance!(g, F, dt, Godunov())

function advance!(g::LevelSetGrid{T}, F::AbstractMatrix, dt, ::Godunov) where {T}
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

#-----------------------------------------------------------------------------# Superbee helpers

# Boundary-aware φ access for wider (2-cell) stencils
@inline function _phi_safe(φ, i, j, ny, nx, ::ZeroNeumann)
    ii = clamp(i, 1, ny)
    jj = clamp(j, 1, nx)
    @inbounds φ[ii, jj]
end

@inline function _phi_safe(φ, i, j, ny, nx, ::Dirichlet)
    ii = clamp(i, 1, ny)
    jj = clamp(j, 1, nx)
    @inbounds φ[ii, jj]
end

@inline function _phi_safe(φ, i, j, ny, nx, ::Periodic)
    ii = mod1(i, ny)
    jj = mod1(j, nx)
    @inbounds φ[ii, jj]
end

# Half-superbee limiter: max(0, max(min(r/2, 1), min(r, 1/2)))
# Equivalent to ELMFIRE's half_superbee function
@inline function _half_superbee(r)
    max(zero(r), max(min(r / 2, one(r)), min(r, one(r) / 2)))
end

# Central-difference front normal with epsilon floor
@inline function _normal_xy(φ, i, j, ny, nx, dx, dy, bc)
    φxp = _phi_safe(φ, i, j + 1, ny, nx, bc)
    φxm = _phi_safe(φ, i, j - 1, ny, nx, bc)
    φyp = _phi_safe(φ, i + 1, j, ny, nx, bc)
    φym = _phi_safe(φ, i - 1, j, ny, nx, bc)
    dφdx = (φxp - φxm) / (2 * dx)
    dφdy = (φyp - φym) / (2 * dy)
    grad = hypot(dφdx, dφdy)
    ε = eps(typeof(grad))
    safe_grad = max(grad, ε)
    return dφdx / safe_grad, dφdy / safe_grad
end

# Superbee flux-limited gradients at cell (i,j)
@inline function _superbee_gradients(φ, ux, uy, i, j, ny, nx, dx, dy, bc, grad_clamp)
    # x-direction
    if ux >= 0
        # upwind stencil: φ[i,j-1], φ[i,j], φ[i,j+1]
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

    # y-direction
    if uy >= 0
        φ_c  = _phi_safe(φ, i, j, ny, nx, bc)
        φ_m  = _phi_safe(φ, i - 1, j, ny, nx, bc)
        φ_mm = _phi_safe(φ, i - 2, j, ny, nx, bc)
        d_main = (φ_c - φ_m) / dy
        d_far  = (φ_m - φ_mm) / dy
        r = abs(d_main) > eps(typeof(d_main)) ? d_far / d_main : zero(d_main)
        ψ = _half_superbee(r)
        dφdy = d_main + ψ * (d_main - d_far)
    else
        φ_c  = _phi_safe(φ, i, j, ny, nx, bc)
        φ_p  = _phi_safe(φ, i + 1, j, ny, nx, bc)
        φ_pp = _phi_safe(φ, i + 2, j, ny, nx, bc)
        d_main = (φ_p - φ_c) / dy
        d_far  = (φ_pp - φ_p) / dy
        r = abs(d_main) > eps(typeof(d_main)) ? d_far / d_main : zero(d_main)
        ψ = _half_superbee(r)
        dφdy = d_main + ψ * (d_main - d_far)
    end

    dφdx = clamp(dφdx, -grad_clamp, grad_clamp)
    dφdy = clamp(dφdy, -grad_clamp, grad_clamp)
    return dφdx, dφdy
end

#-----------------------------------------------------------------------------# Superbee advance!
function advance!(g::LevelSetGrid{T}, F::AbstractMatrix, dt, solver::Superbee) where {T}
    φ = g.φ
    ny, nx = size(φ)
    dx, dy = g.dx, g.dy
    bc = g.bc
    z = zero(T)
    phi_clamp = T(solver.phi_clamp)
    grad_clamp = T(solver.grad_clamp)

    φ_old = copy(φ)
    t_ignite = g.t_ignite
    t_now = g.t + dt

    # Stage 1: Forward Euler
    φ_buf = copy(φ)
    Threads.@threads for j in 1:nx
        for i in 1:ny
            _skip_update(i, j, ny, nx, bc) && continue

            Fij = F[i, j]
            Fij > z || continue

            nx_n, ny_n = _normal_xy(φ_buf, i, j, ny, nx, dx, dy, bc)
            ux = Fij * nx_n
            uy = Fij * ny_n
            dφdx, dφdy = _superbee_gradients(φ_buf, ux, uy, i, j, ny, nx, dx, dy, bc, grad_clamp)

            new_phi = φ_buf[i, j] - dt * (ux * dφdx + uy * dφdy)
            new_phi = isnan(new_phi) ? one(T) : clamp(new_phi, -phi_clamp, phi_clamp)
            φ[i, j] = new_phi
        end
    end

    # Stage 2: Recompute from stage-1 values, then average with φ_old
    φ_buf .= φ  # stage-1 values for consistent reads
    Threads.@threads for j in 1:nx
        for i in 1:ny
            _skip_update(i, j, ny, nx, bc) && continue

            Fij = F[i, j]
            Fij > z || continue

            nx_n, ny_n = _normal_xy(φ_buf, i, j, ny, nx, dx, dy, bc)
            ux = Fij * nx_n
            uy = Fij * ny_n
            dφdx, dφdy = _superbee_gradients(φ_buf, ux, uy, i, j, ny, nx, dx, dy, bc, grad_clamp)

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

#-----------------------------------------------------------------------------# WENO5 helpers

# Core WENO5 reconstruction from 5 consecutive undivided differences.
# Returns the weighted combination of three candidate stencil approximations
# using the Jiang-Shu smoothness indicators and nonlinear weights.
@inline function _weno5_core(v1, v2, v3, v4, v5)
    # Three candidate stencil approximations
    s1 = v1 / 3 - 7v2 / 6 + 11v3 / 6
    s2 = -v2 / 6 + 5v3 / 6 + v4 / 3
    s3 = v3 / 3 + 5v4 / 6 - v5 / 6

    # Smoothness indicators (Jiang & Shu 1996)
    β1 = (13 / 12) * (v1 - 2v2 + v3)^2 + (1 / 4) * (v1 - 4v2 + 3v3)^2
    β2 = (13 / 12) * (v2 - 2v3 + v4)^2 + (1 / 4) * (v2 - v4)^2
    β3 = (13 / 12) * (v3 - 2v4 + v5)^2 + (1 / 4) * (3v3 - 4v4 + v5)^2

    # Nonlinear weights (ideal weights: 1/10, 6/10, 3/10)
    ε = 1e-6
    α1 = oftype(v1, 0.1) / (ε + β1)^2
    α2 = oftype(v1, 0.6) / (ε + β2)^2
    α3 = oftype(v1, 0.3) / (ε + β3)^2
    sum_α = α1 + α2 + α3

    return (α1 * s1 + α2 * s2 + α3 * s3) / sum_α
end

# Left-biased (minus) WENO5 derivative in x-direction at cell (i, j)
@inline function _weno5_minus_x(φ, i, j, ny, nx, dx, bc)
    v1 = _phi_safe(φ, i, j - 2, ny, nx, bc) - _phi_safe(φ, i, j - 3, ny, nx, bc)
    v2 = _phi_safe(φ, i, j - 1, ny, nx, bc) - _phi_safe(φ, i, j - 2, ny, nx, bc)
    v3 = _phi_safe(φ, i, j,     ny, nx, bc) - _phi_safe(φ, i, j - 1, ny, nx, bc)
    v4 = _phi_safe(φ, i, j + 1, ny, nx, bc) - _phi_safe(φ, i, j,     ny, nx, bc)
    v5 = _phi_safe(φ, i, j + 2, ny, nx, bc) - _phi_safe(φ, i, j + 1, ny, nx, bc)
    _weno5_core(v1, v2, v3, v4, v5) / dx
end

# Right-biased (plus) WENO5 derivative in x-direction at cell (i, j)
@inline function _weno5_plus_x(φ, i, j, ny, nx, dx, bc)
    v1 = _phi_safe(φ, i, j + 3, ny, nx, bc) - _phi_safe(φ, i, j + 2, ny, nx, bc)
    v2 = _phi_safe(φ, i, j + 2, ny, nx, bc) - _phi_safe(φ, i, j + 1, ny, nx, bc)
    v3 = _phi_safe(φ, i, j + 1, ny, nx, bc) - _phi_safe(φ, i, j,     ny, nx, bc)
    v4 = _phi_safe(φ, i, j,     ny, nx, bc) - _phi_safe(φ, i, j - 1, ny, nx, bc)
    v5 = _phi_safe(φ, i, j - 1, ny, nx, bc) - _phi_safe(φ, i, j - 2, ny, nx, bc)
    _weno5_core(v1, v2, v3, v4, v5) / dx
end

# Left-biased (minus) WENO5 derivative in y-direction at cell (i, j)
@inline function _weno5_minus_y(φ, i, j, ny, nx, dy, bc)
    v1 = _phi_safe(φ, i - 2, j, ny, nx, bc) - _phi_safe(φ, i - 3, j, ny, nx, bc)
    v2 = _phi_safe(φ, i - 1, j, ny, nx, bc) - _phi_safe(φ, i - 2, j, ny, nx, bc)
    v3 = _phi_safe(φ, i,     j, ny, nx, bc) - _phi_safe(φ, i - 1, j, ny, nx, bc)
    v4 = _phi_safe(φ, i + 1, j, ny, nx, bc) - _phi_safe(φ, i,     j, ny, nx, bc)
    v5 = _phi_safe(φ, i + 2, j, ny, nx, bc) - _phi_safe(φ, i + 1, j, ny, nx, bc)
    _weno5_core(v1, v2, v3, v4, v5) / dy
end

# Right-biased (plus) WENO5 derivative in y-direction at cell (i, j)
@inline function _weno5_plus_y(φ, i, j, ny, nx, dy, bc)
    v1 = _phi_safe(φ, i + 3, j, ny, nx, bc) - _phi_safe(φ, i + 2, j, ny, nx, bc)
    v2 = _phi_safe(φ, i + 2, j, ny, nx, bc) - _phi_safe(φ, i + 1, j, ny, nx, bc)
    v3 = _phi_safe(φ, i + 1, j, ny, nx, bc) - _phi_safe(φ, i,     j, ny, nx, bc)
    v4 = _phi_safe(φ, i,     j, ny, nx, bc) - _phi_safe(φ, i - 1, j, ny, nx, bc)
    v5 = _phi_safe(φ, i - 1, j, ny, nx, bc) - _phi_safe(φ, i - 2, j, ny, nx, bc)
    _weno5_core(v1, v2, v3, v4, v5) / dy
end

# One forward Euler step with WENO5 spatial derivatives:
#   φ_out = φ_in - dt * F * |∇φ_in|_WENO5
function _weno5_rhs!(φ_out, φ_in, F, dt, ny, nx, dx, dy, bc, phi_clamp)
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

            # WENO5 one-sided derivatives
            dxm = _weno5_minus_x(φ_in, i, j, ny, nx, dx, bc)
            dxp = _weno5_plus_x(φ_in, i, j, ny, nx, dx, bc)
            dym = _weno5_minus_y(φ_in, i, j, ny, nx, dy, bc)
            dyp = _weno5_plus_y(φ_in, i, j, ny, nx, dy, bc)

            # Godunov numerical Hamiltonian
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

#-----------------------------------------------------------------------------# WENO5 advance!
function advance!(g::LevelSetGrid{T}, F::AbstractMatrix, dt, solver::WENO5) where {T}
    φ = g.φ
    ny, nx = size(φ)
    dx, dy = g.dx, g.dy
    bc = g.bc
    z = zero(T)
    phi_clamp = T(solver.phi_clamp)

    φ_n = copy(φ)
    φ_1 = similar(φ)
    φ_tmp = similar(φ)

    # SSP-RK3 (Shu-Osher form)
    # Stage 1: φ₁ = φⁿ + dt·L(φⁿ)
    _weno5_rhs!(φ_1, φ_n, F, dt, ny, nx, dx, dy, bc, phi_clamp)

    # Stage 2: φ₂ = ¾φⁿ + ¼(φ₁ + dt·L(φ₁))
    _weno5_rhs!(φ_tmp, φ_1, F, dt, ny, nx, dx, dy, bc, phi_clamp)
    φ_2 = @. T(3 / 4) * φ_n + T(1 / 4) * φ_tmp

    # Stage 3: φⁿ⁺¹ = ⅓φⁿ + ⅔(φ₂ + dt·L(φ₂))
    _weno5_rhs!(φ_tmp, φ_2, F, dt, ny, nx, dx, dy, bc, phi_clamp)
    @. φ = T(1 / 3) * φ_n + T(2 / 3) * φ_tmp

    # Record ignition times
    t_ignite = g.t_ignite
    t_now = g.t + dt
    for j in 1:nx, i in 1:ny
        if φ_n[i, j] >= z && φ[i, j] < z && !isnan(t_ignite[i, j]) && isinf(t_ignite[i, j])
            t_ignite[i, j] = t_now
        end
    end

    g.t = t_now
    g
end

#-----------------------------------------------------------------------------# Curvature

# Mean curvature κ at cell (i,j) via central differences:
#   κ = (φ_xx·φ_y² - 2·φ_x·φ_y·φ_xy + φ_yy·φ_x²) / (φ_x² + φ_y²)^(3/2)
@inline function _curvature(φ, i, j, ny, nx, dx, dy, bc)
    φ_c  = _phi_safe(φ, i, j,     ny, nx, bc)
    φ_xp = _phi_safe(φ, i, j + 1, ny, nx, bc)
    φ_xm = _phi_safe(φ, i, j - 1, ny, nx, bc)
    φ_yp = _phi_safe(φ, i + 1, j, ny, nx, bc)
    φ_ym = _phi_safe(φ, i - 1, j, ny, nx, bc)

    # First derivatives (central)
    φ_x = (φ_xp - φ_xm) / (2 * dx)
    φ_y = (φ_yp - φ_ym) / (2 * dy)

    # Second derivatives
    φ_xx = (φ_xp - 2φ_c + φ_xm) / dx^2
    φ_yy = (φ_yp - 2φ_c + φ_ym) / dy^2

    # Cross derivative
    φ_xpyp = _phi_safe(φ, i + 1, j + 1, ny, nx, bc)
    φ_xpym = _phi_safe(φ, i - 1, j + 1, ny, nx, bc)
    φ_xmyp = _phi_safe(φ, i + 1, j - 1, ny, nx, bc)
    φ_xmym = _phi_safe(φ, i - 1, j - 1, ny, nx, bc)
    φ_xy = (φ_xpyp - φ_xpym - φ_xmyp + φ_xmym) / (4 * dx * dy)

    grad_sq = φ_x^2 + φ_y^2
    grad_mag = sqrt(grad_sq)
    grad_mag < eps(typeof(grad_mag)) && return zero(typeof(grad_mag))
    return (φ_xx * φ_y^2 - 2 * φ_x * φ_y * φ_xy + φ_yy * φ_x^2) / (grad_sq * grad_mag)
end

#-----------------------------------------------------------------------------# cfl_dt
"""
    cfl_dt(grid::LevelSetGrid, F::AbstractMatrix; cfl=0.5, curvature=0.0)

Compute a CFL-stable time step for the level set equation given spread rate field `F`.

Returns `cfl * min(dx, dy) / max(F)`.  If `max(F) ≤ 0` (no active spread), returns `Inf`.

When `curvature > 0`, also enforces the parabolic CFL constraint `dt ≤ dx² / (2b)`.

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
F = fill(10.0, size(grid))
cfl_dt(grid, F)  # 1.5
```
"""
function cfl_dt(g::LevelSetGrid{T}, F::AbstractMatrix; cfl=0.5, curvature=0.0) where {T}
    Fmax = maximum(F)
    Fmax > 0 || return T(Inf)
    dt_adv = T(cfl) * min(g.dx, g.dy) / Fmax
    if curvature > 0
        dt_curv = min(g.dx, g.dy)^2 / (2 * abs(curvature))
        return min(dt_adv, dt_curv)
    end
    dt_adv
end

_smoothed_sign(φ, h) = φ / hypot(φ, h)  # Sussman et al. 1994

#-----------------------------------------------------------------------------# reinitialize!
"""
    reinitialize!(grid::LevelSetGrid; iterations=5)
    reinitialize!(grid::LevelSetGrid, method::AbstractReinitMethod)

Reinitialize `φ` toward a signed distance function.  This prevents `φ` from
becoming too flat or too steep, which degrades gradient accuracy.

Available methods:
- `IterativeReinit(iterations=5)` — PDE-based iterative approach (default)
- `NewtonReinit()` — Newton closest-point for machine-precision accuracy

Should be called periodically (e.g. every 5–10 time steps).
"""
reinitialize!(g::LevelSetGrid; iterations::Int=5) = reinitialize!(g, IterativeReinit(iterations))

function reinitialize!(g::LevelSetGrid{T}, method::IterativeReinit) where {T}
    φ = g.φ
    ny, nx = size(φ)
    dx, dy = g.dx, g.dy
    bc = g.bc
    h = min(dx, dy)
    dτ = h / 2  # pseudo-timestep
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
                dym = _Dym(φ_old, i, j, dy, bc)
                dyp = _Dyp(φ_old, i, j, ny, dy, bc)

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

function reinitialize!(g::LevelSetGrid{T}, method::NewtonReinit) where {T}
    φ = g.φ
    ny, nx = size(φ)
    dx, dy = g.dx, g.dy
    z = zero(T)

    was_nonneg = φ .>= z

    # 1. Build cubic interpolant of φ
    xs_range = xcoords(g)
    ys_range = ycoords(g)
    xs = collect(xs_range)
    ys = collect(ys_range)
    itp = interpolate(collect(φ), BSpline(Cubic(Line(OnGrid()))))
    sitp = extrapolate(scale(itp, ys_range, xs_range), Flat())

    # 2. Find interface cells (sign changes with any neighbor)
    interface_cells = Tuple{Int,Int}[]
    for j in 1:nx, i in 1:ny
        φ_ij = φ[i, j]
        is_iface = false
        j < nx && φ_ij * φ[i, j + 1] < 0 && (is_iface = true)
        i < ny && φ_ij * φ[i + 1, j] < 0 && (is_iface = true)
        j > 1  && φ_ij * φ[i, j - 1] < 0 && (is_iface = true)
        i > 1  && φ_ij * φ[i - 1, j] < 0 && (is_iface = true)
        is_iface && push!(interface_cells, (i, j))
    end

    isempty(interface_cells) && return g

    # 3. Subsample each interface cell and project onto φ=0
    upsample = method.upsample
    maxiters = method.maxiters
    tol = method.tol
    points = SVector{2, T}[]

    for (ic, jc) in interface_cells
        yc = ys[ic]
        xc = xs[jc]
        for di in range(-dy / 2, dy / 2, length=upsample)
            for dj in range(-dx / 2, dx / 2, length=upsample)
                px = xc + dj
                py = yc + di
                # Newton projection onto φ = 0
                for _ in 1:maxiters
                    φ_val = sitp(py, px)
                    abs(φ_val) < tol && break
                    g_yx = Interpolations.gradient(sitp, py, px)
                    gy = g_yx[1]
                    gx = g_yx[2]
                    gsq = gy^2 + gx^2
                    gsq < eps(T) && break
                    px -= φ_val * gx / gsq
                    py -= φ_val * gy / gsq
                end
                abs(sitp(py, px)) < sqrt(tol) && push!(points, SVector{2, T}(px, py))
            end
        end
    end

    isempty(points) && return g

    # 4. Build KD-tree of interface sample points
    tree = KDTree(points)

    # 5. Compute signed distance for each grid point
    for j in 1:nx, i in 1:ny
        x = xs[j]
        y = ys[i]
        query = SVector{2, T}(x, y)
        idx, dist = nn(tree, query)
        cp = points[idx]

        # Refine closest point with Newton
        cpx, cpy = cp[1], cp[2]
        for _ in 1:min(5, maxiters)
            φ_cp = sitp(cpy, cpx)
            abs(φ_cp) < tol && break
            g_yx = Interpolations.gradient(sitp, cpy, cpx)
            gy = g_yx[1]
            gx = g_yx[2]
            gsq = gy^2 + gx^2
            gsq < eps(T) && break
            cpx -= φ_cp * gx / gsq
            cpy -= φ_cp * gy / gsq
        end

        d = hypot(x - cpx, y - cpy)
        φ[i, j] = φ[i, j] >= z ? d : -d
    end

    _update_ignition_after_reinit!(g, was_nonneg)
    g
end

# Shared post-reinit ignition time update
function _update_ignition_after_reinit!(g::LevelSetGrid, was_nonneg)
    φ = g.φ
    t_ignite = g.t_ignite
    t_now = g.t
    z = zero(eltype(φ))
    for j in axes(φ, 2), i in axes(φ, 1)
        if was_nonneg[i, j] && φ[i, j] < z && !isnan(t_ignite[i, j]) && isinf(t_ignite[i, j])
            t_ignite[i, j] = t_now
        end
    end
end

end # module
