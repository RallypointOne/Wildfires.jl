module CellularAutomata

export CellState, UNBURNED, BURNING, BURNED, UNBURNABLE
export AbstractNeighborhood, Moore, VonNeumann
export CAGrid, xcoords, ycoords
export burned, burning, burnable, burn_area
export ignite!, set_unburnable!, advance!

#=
    Cellular Automata Fire Spread Model

    A stochastic, grid-based fire spread model where each cell occupies one of
    four discrete states: UNBURNED, BURNING, BURNED, or UNBURNABLE.

    At each time step:
    1. UNBURNED cells adjacent to BURNING cells may ignite with a probability
       that depends on a user-supplied ignition probability field P[i,j] and a
       distance-based weight (cardinal vs. diagonal neighbors).
    2. BURNING cells whose elapsed burn time exceeds a residence time transition
       to BURNED.

    References:
    - Alexandridis, A., Vakalis, D., Siettos, C.I., & Bafas, G.V. (2008).
      A cellular automata model for forest fire spread prediction. Ecological
      Modelling, 203(3-4), 87–97.
=#

using Random: Random, AbstractRNG, default_rng

#--------------------------------------------------------------------------------# CellState
"""
    CellState

Discrete cell states for the cellular automata fire model.

Values: `UNBURNED`, `BURNING`, `BURNED`, `UNBURNABLE`.
"""
@enum CellState::UInt8 UNBURNED BURNING BURNED UNBURNABLE

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
    CAGrid{T, N} <: AbstractMatrix{CellState}

A 2D grid for cellular automata fire spread simulation.

Each cell has a discrete state (`UNBURNED`, `BURNING`, `BURNED`, `UNBURNABLE`)
and an ignition time (`Inf` = not yet ignited, `NaN` = unburnable).

# Fields
- `state::Matrix{CellState}`   - Per-cell state
- `t_ignite::Matrix{T}`        - Per-cell ignition time [min]
- `dx::T`                      - Grid spacing in x [m]
- `dy::T`                      - Grid spacing in y [m]
- `x0::T`                      - x-coordinate of grid origin [m]
- `y0::T`                      - y-coordinate of grid origin [m]
- `t::T`                       - Current simulation time [min]
- `neighborhood::N`            - Neighborhood type (`Moore()` or `VonNeumann()`)

# Constructor
    CAGrid(nx, ny; dx=30.0, dy=dx, x0=0.0, y0=0.0, neighborhood=Moore())

Create an unburned grid with `nx` columns and `ny` rows.

### Examples
```julia
grid = CAGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)

grid = CAGrid(100, 100, dx=30.0, neighborhood=VonNeumann())
```
"""
mutable struct CAGrid{T, N <: AbstractNeighborhood} <: AbstractMatrix{CellState}
    state::Matrix{CellState}
    t_ignite::Matrix{T}
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
    CAGrid(state, t_ignite, T(dx), T(dy), T(x0), T(y0), zero(T), neighborhood)
end

#--------------------------------------------------------------------------------# AbstractMatrix interface
Base.size(g::CAGrid) = size(g.state)
Base.getindex(g::CAGrid, i...) = getindex(g.state, i...)
Base.setindex!(g::CAGrid, v, i...) = setindex!(g.state, v, i...)

function Base.show(io::IO, ::MIME"text/plain", g::CAGrid{T}) where {T}
    ny, nx = size(g.state)
    nb = count(==(BURNED), g.state)
    nburning = count(==(BURNING), g.state)
    n_unburnable = count(==(UNBURNABLE), g.state)
    neigh_str = g.neighborhood isa Moore ? "" : ", neighborhood=VonNeumann"
    ub_str = n_unburnable > 0 ? ", unburnable=$n_unburnable" : ""
    print(io, "CAGrid{$T} $(nx)×$(ny) (t=$(g.t), burning=$(nburning), burned=$(nb)/$(nx*ny)$(ub_str)$(neigh_str))")
end

function Base.show(io::IO, g::CAGrid{T}) where {T}
    ny, nx = size(g.state)
    print(io, "CAGrid{$T}($(nx)×$(ny))")
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

Return a `BitMatrix` where `true` indicates burned cells (state == `BURNED`).
"""
burned(g::CAGrid) = g.state .== BURNED

"""
    burning(grid::CAGrid)

Return a `BitMatrix` where `true` indicates actively burning cells (state == `BURNING`).
"""
burning(g::CAGrid) = g.state .== BURNING

"""
    burnable(grid::CAGrid)

Return a `BitMatrix` where `true` indicates burnable cells (state != `UNBURNABLE`).
"""
burnable(g::CAGrid) = g.state .!= UNBURNABLE

"""
    burn_area(grid::CAGrid)

Return the total burned area [m²], counting both `BURNING` and `BURNED` cells.
"""
function burn_area(g::CAGrid)
    n = count(s -> s == BURNING || s == BURNED, g.state)
    n * g.dx * g.dy
end

#--------------------------------------------------------------------------------# ignite!
"""
    ignite!(grid::CAGrid, cx, cy, r)

Set a circular ignition at center `(cx, cy)` with radius `r` (all in meters).

Cells within radius `r` that are `UNBURNED` transition to `BURNING` and their
ignition time is recorded.

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
        if hypot(xs[j] - cx, ys[i] - cy) <= r && g.state[i, j] == UNBURNED
            g.state[i, j] = BURNING
            g.t_ignite[i, j] = g.t
        end
    end
    g
end

#--------------------------------------------------------------------------------# set_unburnable!
"""
    set_unburnable!(grid::CAGrid, cx, cy, r)

Mark all cells within radius `r` of `(cx, cy)` as `UNBURNABLE`.

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
            g.state[i, j] = UNBURNABLE
            g.t_ignite[i, j] = eltype(g.t_ignite)(NaN)
        end
    end
    g
end

#--------------------------------------------------------------------------------# advance!
"""
    advance!(grid::CAGrid, P::AbstractMatrix, dt; residence_time=1.0, rng=Random.default_rng())

Advance the CA fire simulation by one time step `dt` [min].

`P[i,j]` is the base ignition probability for cell `(i,j)` per burning neighbor
per time step (values in [0, 1]).

**Spread**: Each `UNBURNED` cell adjacent to a `BURNING` cell rolls independently
for ignition.  The probability is `P[i,j] * w` where `w` is a distance-based
weight (1.0 for cardinal neighbors, `min(dx,dy)/hypot(dx,dy)` for diagonals).

**Burnout**: `BURNING` cells whose elapsed time `≥ residence_time` transition
to `BURNED`.

### Examples
```julia
grid = CAGrid(50, 50, dx=30.0)
ignite!(grid, 750.0, 750.0, 60.0)
P = fill(0.3, size(grid))
advance!(grid, P, 1.0)
```
"""
function advance!(g::CAGrid{T}, P::AbstractMatrix, dt; residence_time=one(T), rng::AbstractRNG=default_rng()) where {T}
    state = g.state
    ny, nx = size(state)
    dx, dy = g.dx, g.dy
    offsets = _offsets(g.neighborhood)
    diag_weight = min(dx, dy) / hypot(dx, dy)

    # Snapshot current state
    old_state = copy(state)
    t_now = g.t + dt

    for j in 1:nx, i in 1:ny
        # Spread: UNBURNED → BURNING
        if old_state[i, j] == UNBURNED
            p_base = P[i, j]
            p_base > 0 || continue
            for (di, dj) in offsets
                ni, nj = i + di, j + dj
                (1 <= ni <= ny && 1 <= nj <= nx) || continue
                old_state[ni, nj] == BURNING || continue
                is_diagonal = di != 0 && dj != 0
                w = is_diagonal ? diag_weight : one(T)
                if rand(rng) < p_base * w
                    state[i, j] = BURNING
                    g.t_ignite[i, j] = t_now
                    break  # cell ignites on first success
                end
            end
        end

        # Burnout: BURNING → BURNED
        if old_state[i, j] == BURNING
            elapsed = t_now - g.t_ignite[i, j]
            if elapsed >= residence_time
                state[i, j] = BURNED
            end
        end
    end

    g.t = t_now
    g
end

end # module
