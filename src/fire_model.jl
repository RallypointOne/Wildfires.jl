using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: topology, xnodes, ynodes, xnode, ynode, Center, Face, Flat
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: launch!

# Fire fields are horizontal-only. `Nothing` in the vertical matches Breeze's
# terrain height field, so terrain and fire state share a field type.
const FireField{G} = Field{Center, Center, Nothing, Nothing, G}

#-----------------------------------------------------------------------------# fire_grid
"""
    fire_grid(atmosphere_grid; refinement = 10, halo = 3)

Build the horizontal grid the fire runs on by refining `atmosphere_grid`'s
horizontal directions by the integer factor `refinement`.

The result covers the same `x`/`y` extent with `Flat` vertical topology, so
every atmospheric cell contains exactly `refinement × refinement` fire cells and
their faces coincide. Exchanging fields between the two grids is then a block
operation rather than an interpolation — aggregating fire fluxes up to the
atmosphere is an exact sum.

`atmosphere_grid` must have uniform horizontal spacing; stretched horizontal
coordinates would not nest exactly. `halo` sets the fire grid's halo, which must
be wide enough for the propagation scheme's stencil.

For a standalone fire simulation with no atmosphere, build the grid directly
with `RectilinearGrid(size = (Nx, Ny), topology = (Bounded, Bounded, Flat), ...)`.

### Examples
```julia
atmosphere = RectilinearGrid(size = (8, 6, 4), x = (-960, 960), y = (-720, 720),
                             z = (0, 400), topology = (Bounded, Bounded, Bounded))
fire_grid(atmosphere; refinement = 8)  # 64×48 cells at 30 m
```
"""
function fire_grid(atmosphere_grid; refinement::Integer = 10, halo::Integer = 3)
    refinement >= 1 || throw(ArgumentError("refinement must be ≥ 1, got $refinement"))

    if !(atmosphere_grid.Δxᶜᵃᵃ isa Number && atmosphere_grid.Δyᵃᶜᵃ isa Number)
        throw(ArgumentError("fire_grid requires uniform horizontal spacing; " *
                            "a stretched atmospheric grid would not nest exactly"))
    end

    TX, TY, _ = topology(atmosphere_grid)
    xf = xnodes(atmosphere_grid, Face())
    yf = ynodes(atmosphere_grid, Face())

    return RectilinearGrid(architecture(atmosphere_grid), eltype(atmosphere_grid);
                           size = (atmosphere_grid.Nx * refinement,
                                   atmosphere_grid.Ny * refinement),
                           halo = (halo, halo),
                           x = (first(xf), last(xf)),
                           y = (first(yf), last(yf)),
                           topology = (TX, TY, Flat))
end

#-----------------------------------------------------------------------------# FireModel
"""
    FireModel(grid; clock = Clock{eltype(grid)}(time = 0))

Fire state on a horizontal `grid`, carrying the two prognostic fields of a
level-set fire spread model:

- `φ` — the level set. `φ < 0` is burning or burned, `φ > 0` is unburned, and
  the fire perimeter is the zero contour.
- `t_ignition` — the time each cell ignited, used to compute post-frontal fuel
  consumption and heat release.

Both are initialized to `Inf`, meaning "no fire anywhere". Establish a fire
before time stepping; advancing an un-ignited model is undefined.

Burned fraction is not stored — it is a function of `t_ignition`, the current
time, and the fuel's burn-out timescale. `scratch` holds the intermediate
Runge–Kutta stage inside [`advance!`](@ref).

### Examples
```julia
model = FireModel(fire_grid(atmosphere; refinement = 8))
```
"""
struct FireModel{G, C, F}
    grid::G
    clock::C
    φ::F
    t_ignition::F
    scratch::F
end

function FireModel(grid; clock = Clock{eltype(grid)}(time = 0))
    topology(grid)[3] === Flat ||
        throw(ArgumentError("FireModel requires a horizontal grid with Flat vertical " *
                            "topology; got $(topology(grid))"))

    FT = eltype(grid)
    φ = Field{Center, Center, Nothing}(grid)
    t_ignition = Field{Center, Center, Nothing}(grid)
    fill!(parent(φ), FT(Inf))
    fill!(parent(t_ignition), FT(Inf))
    scratch = Field{Center, Center, Nothing}(grid)

    return FireModel(grid, clock, φ, t_ignition, scratch)
end

Oceananigans.fields(model::FireModel) = (; φ = model.φ, t_ignition = model.t_ignition)

function Base.show(io::IO, model::FireModel)
    print(io, "FireModel on ", summary(model.grid), "\n",
              "├── clock: ", summary(model.clock), "\n",
              "└── fields: φ, t_ignition")
end

#-----------------------------------------------------------------------------# ignite!
"""
    ignite!(model, x, y; radius, time = model.clock.time)

Ignite a circular fire of `radius` centered at `(x, y)` in grid meters.

`φ` is set to the exact signed distance to that circle — negative inside,
positive outside, zero on the perimeter — so `|∇φ| = 1` and the level set is
well conditioned for advection without a reinitialization pass. Cells inside
the circle get `t_ignition = time`.

Igniting repeatedly takes the pointwise minimum of both fields, which is the
signed distance to the *union* of the circles and the earliest ignition time of
each cell. So several ignitions compose, in any order.

### Examples
```julia
model = FireModel(fire_grid(atmosphere; refinement = 8))
ignite!(model, 0.0, 0.0; radius = 100.0)

# a second ignition 500 m east; φ is now the distance to whichever is nearer
ignite!(model, 500.0, 0.0; radius = 50.0)
```
"""
function ignite!(model::FireModel, x, y; radius, time = model.clock.time)
    radius > 0 || throw(ArgumentError("radius must be positive, got $radius"))

    grid = model.grid
    FT = eltype(grid)
    launch!(architecture(grid), grid, :xy, _ignite!, model.φ, model.t_ignition, grid,
            FT(x), FT(y), FT(radius), FT(time))

    fill_halo_regions!(model.φ)
    fill_halo_regions!(model.t_ignition)

    return model
end

# The distance's derivative is undefined at a cell center that coincides
# exactly with (x₀, y₀); centers sit at half-cell offsets, so this only
# happens when an ignition point is placed there deliberately.
@kernel function _ignite!(φ, t_ignition, grid, x₀, y₀, radius, time)
    i, j = @index(Global, NTuple)
    x = xnode(i, j, 1, grid, Center(), Center(), Center())
    y = ynode(i, j, 1, grid, Center(), Center(), Center())
    d = sqrt((x - x₀)^2 + (y - y₀)^2) - radius
    @inbounds begin
        φ[i, j, 1] = min(φ[i, j, 1], d)
        t_ignition[i, j, 1] = ifelse(d < 0, min(t_ignition[i, j, 1], time), t_ignition[i, j, 1])
    end
end

#-----------------------------------------------------------------------------# advance!
"""
    advance!(model, Δt; speed)

Advance the fire front by `Δt` seconds. `speed` is the rate of spread in m/s,
either a number or a field on `model.grid`, and must be non-negative.

Solves the level-set equation `∂φ/∂t + speed |∇φ| = 0` with the Godunov upwind
Hamiltonian and Heun's method (second-order Runge–Kutta), the scheme used by
WRF-SFIRE. The stencil needs a halo of one cell. Stable when
`Δt * speed * (1/Δx + 1/Δy) ≤ 1`; nothing enforces this.

A cell whose `φ` crosses zero during the step gets `t_ignition` from linear
interpolation of the crossing time within the step. The clock advances by `Δt`.

### Examples
```julia
model = FireModel(grid)
ignite!(model, 0.0, 0.0; radius = 100.0)
for _ in 1:100
    advance!(model, 1.0; speed = 2.0)
end
```
"""
function advance!(model::FireModel, Δt; speed)
    grid = model.grid
    FT = eltype(grid)
    arch = architecture(grid)
    φ, t_ignition, φ¹ = model.φ, model.t_ignition, model.scratch
    speed = speed isa Number ? FT(speed) : speed

    launch!(arch, grid, :xy, _heun_stage1!, φ¹, φ, speed, grid, FT(Δt))
    fill_halo_regions!(φ¹)

    launch!(arch, grid, :xy, _heun_stage2!, φ, t_ignition, φ¹, speed, grid,
            FT(Δt), FT(model.clock.time))
    fill_halo_regions!(φ)
    fill_halo_regions!(t_ignition)

    tick!(model.clock, Δt)
    return model
end

@inline _at(speed::Number, i, j) = speed
@inline _at(speed, i, j) = @inbounds speed[i, j, 1]

# Godunov upwind |∇φ| for non-negative speed (Osher & Fedkiw 2003, §6.2):
# per direction, the larger of the backward difference's positive part and the
# forward difference's negative part.
@inline function _godunov_∇φ(i, j, grid, φ)
    z = zero(eltype(grid))
    D⁻ₓ = ∂xᶠᶜᶜ(i, j, 1, grid, φ)
    D⁺ₓ = ∂xᶠᶜᶜ(i + 1, j, 1, grid, φ)
    D⁻ᵧ = ∂yᶜᶠᶜ(i, j, 1, grid, φ)
    D⁺ᵧ = ∂yᶜᶠᶜ(i, j + 1, 1, grid, φ)
    return sqrt(max(max(D⁻ₓ, z), -min(D⁺ₓ, z))^2 + max(max(D⁻ᵧ, z), -min(D⁺ᵧ, z))^2)
end

@kernel function _heun_stage1!(φ¹, φ, speed, grid, Δt)
    i, j = @index(Global, NTuple)
    @inbounds φ¹[i, j, 1] = φ[i, j, 1] - Δt * _at(speed, i, j) * _godunov_∇φ(i, j, grid, φ)
end

# Reads φ only at (i, j), so writing φ in place is safe.
@kernel function _heun_stage2!(φ, t_ignition, φ¹, speed, grid, Δt, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        φⁿ = φ[i, j, 1]
        φⁿ⁺¹ = (φⁿ + φ¹[i, j, 1] - Δt * _at(speed, i, j) * _godunov_∇φ(i, j, grid, φ¹)) / 2
        crossed = (φⁿ >= 0) & (φⁿ⁺¹ < 0)
        Δφ = ifelse(crossed, φⁿ - φⁿ⁺¹, one(φⁿ))    # keeps the unselected branch finite
        t_ignition[i, j, 1] = ifelse(crossed, time + Δt * φⁿ / Δφ, t_ignition[i, j, 1])
        φ[i, j, 1] = φⁿ⁺¹
    end
end
