module CellularAutomata

export CellState, UNBURNED, BURNING, BURNED, UNBURNABLE
export AbstractNeighborhood, Moore, VonNeumann
export CAGrid, xcoords, ycoords
export burned, burning, burnable, burn_area
export ignite!, set_unburnable!, advance!
export cellstate, on_ignite, on_burnout, on_unburnable

#=
    Cellular Automata Fire Spread Model (Deterministic Travel-Time)

    A deterministic, grid-based fire spread model where each cell holds a value
    of type `S` (default `CellState`).  Users can parameterize `CAGrid{T, S, N}`
    with custom cell types by implementing the cell interface:

        cellstate(cell)        -> CellState
        on_ignite(cell, t)     -> cell
        on_burnout(cell, t)    -> cell
        on_unburnable(cell)    -> cell

    Fire propagates via a travel-time approach (Cell2Fire-style):
    - For each BURNING cell, compute directional spread rate R(θ) to each neighbor
    - Travel time = distance / R(θ)
    - Cell ignites when the minimum arrival time from any burning neighbor is reached
    - BURNING cells transition to BURNED after a residence time

    References:
    - Carrasco, J. et al. (2021). Cell2Fire: A cell-based forest fire growth model.
      Frontiers in Forests and Global Change, 4, 692706.
=#

#--------------------------------------------------------------------------------# CellState
"""
    CellState

Discrete cell states for the cellular automata fire model.

Values: `UNBURNED`, `BURNING`, `BURNED`, `UNBURNABLE`.
"""
@enum CellState::UInt8 UNBURNED BURNING BURNED UNBURNABLE

#--------------------------------------------------------------------------------# Cell Interface
"""
    cellstate(cell) -> CellState

Return the fire state of a cell.  Must be implemented for custom cell types
used with [`CAGrid`](@ref).

Default: identity for `CellState`.
"""
cellstate(s::CellState) = s

"""
    on_ignite(cell, t) -> cell

Return a new cell representing the ignited state at time `t`.
Must be implemented for custom cell types used with [`CAGrid`](@ref).

Default: returns `BURNING` for `CellState`.
"""
on_ignite(::CellState, t) = BURNING

"""
    on_burnout(cell, t) -> cell

Return a new cell representing the burned-out state at time `t`.
Must be implemented for custom cell types used with [`CAGrid`](@ref).

Default: returns `BURNED` for `CellState`.
"""
on_burnout(::CellState, t) = BURNED

"""
    on_unburnable(cell) -> cell

Return a new cell representing an unburnable state.
Must be implemented for custom cell types used with [`CAGrid`](@ref).

Default: returns `UNBURNABLE` for `CellState`.
"""
on_unburnable(::CellState) = UNBURNABLE

#--------------------------------------------------------------------------------# Neighborhoods
"""
    AbstractNeighborhood

Supertype for cellular automata neighborhood definitions.

Subtypes: [`Moore`](@ref), [`VonNeumann`](@ref).
"""
abstract type AbstractNeighborhood end

"""
    Moore()

8-connected neighborhood (cardinal + diagonal neighbors).

### Examples
```julia
grid = CAGrid(50, 50, neighborhood=Moore())
```
"""
struct Moore <: AbstractNeighborhood end

"""
    VonNeumann()

4-connected neighborhood (cardinal neighbors only).

### Examples
```julia
grid = CAGrid(50, 50, neighborhood=VonNeumann())
```
"""
struct VonNeumann <: AbstractNeighborhood end

# Neighbor offsets: (di, dj)
_offsets(::Moore) = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_offsets(::VonNeumann) = ((-1, 0), (0, -1), (0, 1), (1, 0))

#--------------------------------------------------------------------------------# CAGrid
"""
    CAGrid{T, S, N} <: AbstractMatrix{S}

A 2D grid for cellular automata fire spread simulation, parameterized on cell
type `S`.

The default cell type is [`CellState`](@ref).  For custom cell types, implement
the cell interface: [`cellstate`](@ref), [`on_ignite`](@ref), [`on_burnout`](@ref),
and optionally [`on_unburnable`](@ref).

# Fields
- `state::Matrix{S}`            - Per-cell data
- `t_ignite::Matrix{T}`         - Per-cell ignition time [min]
- `t_arrival::Matrix{T}`        - Per-cell minimum pending arrival time [min]
- `dx::T`, `dy::T`              - Grid spacing [m]
- `x0::T`, `y0::T`              - Grid origin [m]
- `t::T`                        - Current simulation time [min]
- `neighborhood::N`             - Neighborhood type (`Moore()` or `VonNeumann()`)

# Constructors
    CAGrid(nx, ny; dx=30.0, dy=dx, x0=0.0, y0=0.0, neighborhood=Moore())
    CAGrid(cells::Matrix{S}; dx=30.0, dy=dx, x0=0.0, y0=0.0, neighborhood=Moore())

### Examples
```julia
grid = CAGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)

# Custom cell type
cells = fill(MyCell(...), 100, 100)
grid = CAGrid(cells, dx=30.0)
```
"""
mutable struct CAGrid{T, S, N <: AbstractNeighborhood} <: AbstractMatrix{S}
    state::Matrix{S}
    t_ignite::Matrix{T}
    t_arrival::Matrix{T}
    dx::T
    dy::T
    x0::T
    y0::T
    t::T
    neighborhood::N
end

function CAGrid(nx::Integer, ny::Integer; dx=30.0, dy=dx, x0=0.0, y0=0.0, neighborhood::AbstractNeighborhood=Moore())
    T = promote_type(typeof(dx), typeof(dy), typeof(x0), typeof(y0))
    state = fill(UNBURNED, ny, nx)
    t_ignite = fill(T(Inf), ny, nx)
    t_arrival = fill(T(Inf), ny, nx)
    CAGrid(state, t_ignite, t_arrival, T(dx), T(dy), T(x0), T(y0), zero(T), neighborhood)
end

function CAGrid(cells::Matrix{S}; dx=30.0, dy=dx, x0=0.0, y0=0.0, neighborhood::AbstractNeighborhood=Moore()) where {S}
    T = promote_type(typeof(dx), typeof(dy), typeof(x0), typeof(y0))
    ny, nx = size(cells)
    t_ignite = fill(T(Inf), ny, nx)
    t_arrival = fill(T(Inf), ny, nx)
    for j in 1:nx, i in 1:ny
        cs = cellstate(cells[i, j])
        if cs == BURNING
            t_ignite[i, j] = zero(T)
        elseif cs == UNBURNABLE
            t_ignite[i, j] = T(NaN)
        end
    end
    CAGrid(cells, t_ignite, t_arrival, T(dx), T(dy), T(x0), T(y0), zero(T), neighborhood)
end

#--------------------------------------------------------------------------------# AbstractMatrix interface
Base.size(g::CAGrid) = size(g.state)
Base.getindex(g::CAGrid, i...) = getindex(g.state, i...)
Base.setindex!(g::CAGrid, v, i...) = setindex!(g.state, v, i...)

function Base.show(io::IO, ::MIME"text/plain", g::CAGrid{T, S}) where {T, S}
    ny, nx = size(g.state)
    nb = count(c -> cellstate(c) == BURNED, g.state)
    nburning = count(c -> cellstate(c) == BURNING, g.state)
    n_unburnable = count(c -> cellstate(c) == UNBURNABLE, g.state)
    neigh_str = g.neighborhood isa Moore ? "" : ", neighborhood=VonNeumann"
    ub_str = n_unburnable > 0 ? ", unburnable=$n_unburnable" : ""
    cell_str = S === CellState ? "" : ", cell=$S"
    print(io, "CAGrid{$T} $(nx)×$(ny) (t=$(g.t), burning=$(nburning), burned=$(nb)/$(nx*ny)$(ub_str)$(cell_str)$(neigh_str))")
end

function Base.show(io::IO, g::CAGrid{T, S}) where {T, S}
    ny, nx = size(g.state)
    cell_str = S === CellState ? "" : ", $S"
    print(io, "CAGrid{$T$(cell_str)}($(nx)×$(ny))")
end

#--------------------------------------------------------------------------------# Coordinates
"""
    xcoords(grid::CAGrid)

Return the x-coordinates of grid cell centers.
"""
xcoords(g::CAGrid) = range(g.x0 + g.dx / 2, step=g.dx, length=size(g.state, 2))

"""
    ycoords(grid::CAGrid)

Return the y-coordinates of grid cell centers.
"""
ycoords(g::CAGrid) = range(g.y0 + g.dy / 2, step=g.dy, length=size(g.state, 1))

#--------------------------------------------------------------------------------# Queries
"""
    burned(grid::CAGrid)

Return a `BitMatrix` where `true` indicates burned cells.
"""
burned(g::CAGrid) = map(c -> cellstate(c) == BURNED, g.state)

"""
    burning(grid::CAGrid)

Return a `BitMatrix` where `true` indicates actively burning cells.
"""
burning(g::CAGrid) = map(c -> cellstate(c) == BURNING, g.state)

"""
    burnable(grid::CAGrid)

Return a `BitMatrix` where `true` indicates burnable cells.
"""
burnable(g::CAGrid) = map(c -> cellstate(c) != UNBURNABLE, g.state)

"""
    burn_area(grid::CAGrid)

Return the total burned area [m²], counting both `BURNING` and `BURNED` cells.
"""
function burn_area(g::CAGrid)
    n = count(g.state) do c
        s = cellstate(c)
        s == BURNING || s == BURNED
    end
    n * g.dx * g.dy
end

#--------------------------------------------------------------------------------# ignite!
"""
    ignite!(grid::CAGrid, cx, cy, r)

Set a circular ignition at center `(cx, cy)` with radius `r` (all in meters).

Cells within radius `r` that are `UNBURNED` transition via [`on_ignite`](@ref)
and their ignition time is recorded.

### Examples
```julia
grid = CAGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)
```
"""
function ignite!(g::CAGrid, cx, cy, r)
    xs = xcoords(g)
    ys = ycoords(g)
    for j in eachindex(xs), i in eachindex(ys)
        if hypot(xs[j] - cx, ys[i] - cy) <= r && cellstate(g.state[i, j]) == UNBURNED
            g.state[i, j] = on_ignite(g.state[i, j], g.t)
            g.t_ignite[i, j] = g.t
        end
    end
    g
end

#--------------------------------------------------------------------------------# set_unburnable!
"""
    set_unburnable!(grid::CAGrid, cx, cy, r)

Mark all cells within radius `r` of `(cx, cy)` as unburnable via [`on_unburnable`](@ref).

### Examples
```julia
grid = CAGrid(100, 100, dx=30.0)
set_unburnable!(grid, 1500.0, 1500.0, 200.0)
```
"""
function set_unburnable!(g::CAGrid, cx, cy, r)
    xs = xcoords(g)
    ys = ycoords(g)
    for j in eachindex(xs), i in eachindex(ys)
        if hypot(xs[j] - cx, ys[i] - cy) <= r
            g.state[i, j] = on_unburnable(g.state[i, j])
            g.t_ignite[i, j] = eltype(g.t_ignite)(NaN)
        end
    end
    g
end

#--------------------------------------------------------------------------------# advance! (stub — model-aware method defined in SpreadModels)
"""
    advance!(grid::CAGrid, model, dt; burnout, burnin, residence_time)

Advance the CA fire simulation by one time step `dt` [min] using a travel-time
approach with directional spread rates from `model`.

The model-aware method is defined in `SpreadModels` — see [`simulate!`](@ref) for
the recommended high-level interface.
"""
function advance! end

end # module
