module LevelSet

export LevelSetGrid, xcoords, ycoords, burned, burn_area, ignite!, advance!, reinitialize!
export fireplot, fireplot!

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
using Makie

#-----------------------------------------------------------------------------# LevelSetGrid
"""
    LevelSetGrid{T} <: AbstractMatrix{T}

A 2D grid representing a fire spread simulation via the level set method.

The grid stores a signed distance function `φ` where:
- `φ < 0` → burned
- `φ = 0` → fire front
- `φ > 0` → unburned

# Fields
- `φ::Matrix{T}`  - Level set function values
- `dx::T`         - Grid spacing in x [m]
- `dy::T`         - Grid spacing in y [m]
- `x0::T`         - x-coordinate of grid origin (lower-left) [m]
- `y0::T`         - y-coordinate of grid origin (lower-left) [m]
- `t::T`          - Current simulation time [min]

# Constructor
    LevelSetGrid(nx, ny; dx=30.0, dy=dx, x0=0.0, y0=0.0)

Create an unburned grid with `nx × ny` cells.

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)
```
"""
mutable struct LevelSetGrid{T} <: AbstractMatrix{T}
    φ::Matrix{T}
    dx::T
    dy::T
    x0::T
    y0::T
    t::T
end

function LevelSetGrid(nx::Integer, ny::Integer; dx=30.0, dy=dx, x0=0.0, y0=0.0)
    T = promote_type(typeof(dx), typeof(dy), typeof(x0), typeof(y0))
    φ = ones(T, ny, nx)  # all positive → unburned
    LevelSetGrid(φ, T(dx), T(dy), T(x0), T(y0), zero(T))
end

#-----------------------------------------------------------------------------# AbstractMatrix interface
Base.size(g::LevelSetGrid) = size(g.φ)
Base.getindex(g::LevelSetGrid, i...) = getindex(g.φ, i...)
Base.setindex!(g::LevelSetGrid, v, i...) = setindex!(g.φ, v, i...)

function Base.show(io::IO, ::MIME"text/plain", g::LevelSetGrid{T}) where {T}
    ny, nx = size(g.φ)
    nb = count(<(0), g.φ)
    print(io, "LevelSetGrid{$T} $(nx)×$(ny) (t=$(g.t), burned=$(nb)/$(nx*ny))")
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
        g.φ[i, j] = min(g.φ[i, j], d)
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
    z = zero(T)

    # In-place update (safe because each cell only reads neighbors of φ)
    # We need a copy of φ to read from while writing
    φ_old = copy(φ)

    for j in 1:nx, i in 1:ny
        Fij = F[i, j]
        Fij > z || continue  # no spread where F ≤ 0

        # --- Upwind finite differences (Godunov) ---
        # x-direction
        Dxm = j > 1  ? (φ_old[i, j] - φ_old[i, j-1]) / dx : z
        Dxp = j < nx ? (φ_old[i, j+1] - φ_old[i, j]) / dx : z

        # y-direction
        Dym = i > 1  ? (φ_old[i, j] - φ_old[i-1, j]) / dy : z
        Dyp = i < ny ? (φ_old[i+1, j] - φ_old[i, j]) / dy : z

        # Godunov upwind: for F > 0, use max of backward and min of forward
        Dxm_plus = max(Dxm, z)
        Dxp_minus = min(Dxp, z)
        Dym_plus = max(Dym, z)
        Dyp_minus = min(Dyp, z)

        grad_sq = max(Dxm_plus, -Dxp_minus)^2 + max(Dym_plus, -Dyp_minus)^2
        grad_sq > z || continue  # no gradient → no front to propagate
        grad_mag = sqrt(grad_sq)

        φ[i, j] = φ_old[i, j] - dt * Fij * grad_mag
    end

    g.t += dt
    g
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
    dτ = min(dx, dy) * 0.5  # pseudo-timestep
    z = zero(T)

    for _ in 1:iterations
        φ_old = copy(φ)
        for j in 1:nx, i in 1:ny
            S = sign(φ_old[i, j])

            # Central differences for gradient
            Dxm = j > 1  ? (φ_old[i, j] - φ_old[i, j-1]) / dx : z
            Dxp = j < nx ? (φ_old[i, j+1] - φ_old[i, j]) / dx : z
            Dym = i > 1  ? (φ_old[i, j] - φ_old[i-1, j]) / dy : z
            Dyp = i < ny ? (φ_old[i+1, j] - φ_old[i, j]) / dy : z

            if S > 0
                a = max(max(Dxm, z), -min(Dxp, z))
                b = max(max(Dym, z), -min(Dyp, z))
            else
                a = max(-min(Dxm, z), max(Dxp, z))
                b = max(-min(Dym, z), max(Dyp, z))
            end

            grad_mag = hypot(a, b)
            φ[i, j] = φ_old[i, j] - dτ * S * (grad_mag - one(T))
        end
    end
    g
end

#-----------------------------------------------------------------------------# Makie recipes
"""
    heatmap(grid::LevelSetGrid)
    contour(grid::LevelSetGrid)
    contourf(grid::LevelSetGrid)
    surface(grid::LevelSetGrid)

Standard Makie plot types work on `LevelSetGrid` with correct spatial coordinates.
The plotted values are the level set function `φ`.
"""
function Makie.convert_arguments(P::Type{<:Union{Makie.Heatmap,Makie.Contour,Makie.Contourf,Makie.Surface}}, g::LevelSetGrid)
    Makie.convert_arguments(P, collect(xcoords(g)), collect(ycoords(g)), copy(g.φ))
end

"""
    fireplot(grid::LevelSetGrid; colormap=:RdYlGn, frontcolor=:black, frontlinewidth=2.0)
    fireplot!(ax, grid; ...)

Plot a `LevelSetGrid` as a heatmap of the `φ` field with the fire front (`φ = 0`)
overlaid as a contour line.

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)
fireplot(grid)
```
"""
@recipe(FirePlot, grid) do scene
    Attributes(
        colormap = :RdYlGn,
        frontcolor = :black,
        frontlinewidth = 2.0,
    )
end

function Makie.plot!(p::FirePlot)
    grid_obs = p[1]
    xs = @lift collect(xcoords($grid_obs))
    ys = @lift collect(ycoords($grid_obs))
    φ = @lift copy($grid_obs.φ)

    heatmap!(p, xs, ys, φ; colormap=p[:colormap])
    contour!(p, xs, ys, φ; levels=[0.0], color=p[:frontcolor], linewidth=p[:frontlinewidth])
    p
end

end # module
