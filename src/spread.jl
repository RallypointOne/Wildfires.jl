using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ
using Oceananigans.OutputReaders: FlavorOfFTS, Time

#-----------------------------------------------------------------------------# wind_adjustment
"""
    wind_adjustment(bed::FuelBed)

Albini and Baughman (1979) unsheltered wind adjustment factor: the ratio of
midflame wind to the 20 ft open wind for a fuel bed of depth `bed.fuel.δ`
ft. 0.36 for the 1 ft beds of fuel models 1 and 2.

### Examples
```julia
wind_adjustment(FuelBed(Wildfires.NFFL.SHORT_GRASS))   # ≈ 0.36
```
"""
function wind_adjustment(bed::FuelBed{T}) where {T}
    δ = _pos(bed.fuel.δ)
    return T(1.83) / log((20 + T(0.36) * δ) / (T(0.13) * δ))
end

#-----------------------------------------------------------------------------# length_to_breadth
"""
    length_to_breadth(midflame_wind; max_length_to_breadth = 8)

Length-to-breadth ratio of the elliptical fire shape from the midflame wind
[m/s], Anderson (1983) as used in FARSITE and ELMFIRE:
`0.936 e^{0.2566 U} + 0.461 e^{-0.1548 U} - 0.397` with `U` in mi/h, at least
1 and at most `max_length_to_breadth`.

### Examples
```julia
length_to_breadth(0.0)   # 1.0
length_to_breadth(2.0)   # ≈ 2.7
```
"""
function length_to_breadth(midflame_wind; max_length_to_breadth = 8)
    U = max(midflame_wind, zero(midflame_wind)) * oftype(float(midflame_wind), 2.23694)
    LB = oftype(U, 0.936) * exp(oftype(U, 0.2566) * U) +
         oftype(U, 0.461) * exp(oftype(U, -0.1548) * U) - oftype(U, 0.397)
    return clamp(LB, one(U), oftype(U, max_length_to_breadth))
end

#-----------------------------------------------------------------------------# Front shapes
"""
    NormalProjection()

Front-shape option for [`spread_rate!`](@ref): WRF-SFIRE's formulation. Wind
and slope are resolved onto the outward normal and a negative component is
treated as calm or flat, so the front spreads at the head-fire rate downwind
and upslope and at the no-wind, no-slope rate against them. Flanks spread at
the no-wind rate, which is a small fraction of the head rate; SFIRE relies on
fire-induced winds from the coupled atmosphere to widen the fire.
"""
struct NormalProjection end

"""
    HuygensEllipse(; max_length_to_breadth = 8)

Front-shape option for [`spread_rate!`](@ref), the default: the FARSITE and
ELMFIRE formulation. The slope is converted to the wind that would give the
same Rothermel slope factor and added to the midflame wind vector; the head
rate is the Rothermel rate in that effective wind, the length-to-breadth
ratio comes from [`length_to_breadth`](@ref), and the normal speed is the
support function of that ellipse ([`ellipse_speed`](@ref)). Flanks spread at
head over roughly the length-to-breadth ratio.
"""
struct HuygensEllipse{T}
    max_length_to_breadth::T
end
HuygensEllipse(; max_length_to_breadth = 8.0) = HuygensEllipse(max_length_to_breadth)

#-----------------------------------------------------------------------------# FuelMap
"""
    FuelMap(grid, code, table; remap = ())

Per-cell fuel for [`spread_rate!`](@ref) on a horizontal `grid`: `code` is a
function `(x, y) -> code` (for instance from [`raster_sampler`](@ref) with
`method = :near`) or a field of codes, and `table` is `Wildfires.NFFL`,
`Wildfires.SB40`, or any collection of [`FuelModel`](@ref)s. Every model in
`table` becomes a [`FuelBed`](@ref) on the grid's architecture, and each cell
stores the index of the bed whose `code` it holds. Codes absent from `table`
(LANDFIRE's non-burnable 91–99, or 0 off the raster) map to
`Wildfires.NONBURNABLE`, which spreads at zero.

`remap` is a collection of `from => to` code pairs applied before the lookup.
Its main use is the wildland-urban interface: LANDFIRE classes developed land
as 91, non-burnable, which stops the vegetation front at every street even
though lawns and ornamental plantings carry fire between houses. WRF-Fire
case studies of the Camp and Marshall fires substitute a grass model there;
`remap = (91 => 1,)` does the same with short grass.

### Examples
```julia
fuel = FuelMap(grid, raster_sampler(Raster("fuel.tif"), crs; method = :near), Wildfires.NFFL;
               remap = (91 => 1,))
spread_rate!(speed, model, fuel, moisture; wind = (u, v))
```
"""
struct FuelMap{F, B}
    index::F
    beds::B
    codes::Vector{Int}
end

function FuelMap(grid, code, table; remap = ())
    FT = eltype(grid)
    models = collect(FuelModel, table isa NamedTuple ? values(table) : table)
    push!(models, NONBURNABLE)
    codes = [m.code for m in models]
    allunique(codes[1:end-1]) || throw(ArgumentError("fuel table has duplicate codes"))
    lookup = Dict(c => k for (k, c) in enumerate(codes))
    nonburnable = length(models)
    remap = Dict{Int, Int}(remap)
    for (from, to) in remap
        haskey(lookup, to) || throw(ArgumentError("remap target $to for code $from is not in the fuel table"))
    end
    index_of(c) = (k = round(Int, c); FT(get(lookup, get(remap, k, k), nonburnable)))

    index = Field{Center, Center, Nothing}(grid)
    if code isa Function
        set!(index, (x, y) -> index_of(code(x, y)))
    else
        set!(index, index_of.(Array(interior(code))))
    end
    beds = on_architecture(architecture(grid), [FuelBed{FT}(m) for m in models])
    return FuelMap(index, beds, codes)
end

Base.show(io::IO, f::FuelMap) =
    print(io, "FuelMap: ", length(f.codes) - 1, " fuel models on ", summary(f.index.grid))

#-----------------------------------------------------------------------------# spread_rate!
"""
    spread_rate!(speed, model::FireModel, fuel, moisture::FuelClasses;
                 wind = (0, 0), slope = (0, 0), shape = HuygensEllipse(),
                 structures = nothing, hamada = HamadaModel())

Fill `speed`, a field on `model.grid`, with the rate of spread [m/s] normal
to the fire front in every cell, for use as the `speed` argument of
[`advance!`](@ref).

`fuel` is one [`FuelBed`](@ref) for the whole domain or a [`FuelMap`](@ref).
`wind` is the open 20 ft wind `(u, v)` [m/s] and `slope` the terrain gradient
`(∂h/∂x, ∂h/∂y)` as rise over run; each component is a number, a field on
`model.grid`, or a `FieldTimeSeries` on it, read at `model.clock.time`.
Vegetation spreads at the Rothermel rate of the cell's bed with the wind
reduced to midflame height by [`wind_adjustment`](@ref), shaped by `shape`:
[`HuygensEllipse`](@ref) (default) or [`NormalProjection`](@ref). Where `∇φ`
vanishes the normal is undefined and the no-wind rate is used; that happens
only away from the front.

With `structures`, cells within their neighbourhood also spread at the Hamada
urban rate — [`hamada_rates`](@ref) from the open wind and the structure
fields, shaped by [`ellipse_speed`](@ref) — and the cell takes the larger of
the two rates.

`moisture` applies to every cell. Recompute after each [`advance!`](@ref),
since the normal changes as the front moves.

### Examples
```julia
speed = Field{Center, Center, Nothing}(model.grid)
bed = FuelBed(Wildfires.NFFL.SHORT_GRASS)
for _ in 1:100
    spread_rate!(speed, model, bed, FuelClasses(0.08, 1.0); wind = (5.0, 0.0))
    advance!(model, 1.0; speed)
end
```
"""
function spread_rate!(speed, model::FireModel, fuel::Union{FuelBed, FuelMap}, moisture::FuelClasses;
                      wind = (0, 0), slope = (0, 0), shape = HuygensEllipse(),
                      structures = nothing, hamada = HamadaModel())
    grid = model.grid
    FT = eltype(grid)
    u, v = _component(wind[1], FT), _component(wind[2], FT)
    ∂x_h, ∂y_h = _component(slope[1], FT), _component(slope[2], FT)

    launch!(architecture(grid), grid, :xy, _spread_rate!, speed, model.φ, grid,
            _fuel_arg(fuel, FT), FuelClasses{FT}(moisture), _shape_arg(shape, FT),
            u, v, ∂x_h, ∂y_h, _urban_fields(structures), HamadaModel{FT}(hamada),
            FT(model.clock.time))
    fill_halo_regions!(speed)

    return speed
end

_component(c::Number, FT) = FT(c)
_component(c, FT) = c

_fuel_arg(bed::FuelBed, FT) = FuelBed{FT}(bed)
_fuel_arg(fuel::FuelMap, FT) = (; index = fuel.index, beds = fuel.beds)

_shape_arg(::NormalProjection, FT) = NormalProjection()
_shape_arg(s::HuygensEllipse, FT) = HuygensEllipse(FT(s.max_length_to_breadth))

_urban_fields(::Nothing) = nothing
_urban_fields(s::Structures) = (; plan = s.plan_dimension, separation = s.separation,
                                  nonburnable = s.nonburnable_fraction)

# A number, a field, or a time series read at time `t`.
@inline _at(w::Number, i, j, t) = w
@inline _at(w::FlavorOfFTS, i, j, t) = @inbounds w[i, j, 1, Time(t)]
@inline _at(w, i, j, t) = @inbounds w[i, j, 1]

@inline _bed(bed::FuelBed, i, j) = bed
@inline _bed(fuel, i, j) = @inbounds fuel.beds[Int(fuel.index[i, j, 1])]

# Wind [m/s] that gives the same Rothermel factor as slope `s` (rise over run).
@inline function _slope_wind(bed::FuelBed{T}, s) where {T}
    φₛ = bed.C_s * s^2
    U = (φₛ / _pos(bed.C_w))^(1 / _pos(bed.B))   # ft/min
    return U / T(3.28084 * 60)
end

@inline function _vegetation_speed(::NormalProjection, bed, moisture, nx, ny, u, v, hx, hy)
    wind_n = wind_adjustment(bed) * (u * nx + v * ny)
    slope_n = hx * nx + hy * ny
    return spread_rate(bed, moisture, wind_n, slope_n)
end

@inline function _vegetation_speed(shape::HuygensEllipse, bed, moisture, nx, ny, u, v, hx, hy)
    waf = wind_adjustment(bed)
    s = sqrt(hx^2 + hy^2)
    Uₛ = _slope_wind(bed, s)
    ex = waf * u + Uₛ * hx / _pos(s)             # effective midflame wind vector
    ey = waf * v + Uₛ * hy / _pos(s)
    U = sqrt(ex^2 + ey^2)
    wx, wy = ex / _pos(U), ey / _pos(U)
    head = spread_rate(bed, moisture, U, zero(U))
    LB = length_to_breadth(U; max_length_to_breadth = shape.max_length_to_breadth)
    e = sqrt(max(1 - 1 / LB^2, zero(U)))
    a = head / (1 + e)
    return ellipse_speed(nx, ny, wx, wy, head, a / LB, a * (1 - e))
end

@inline _urban_speed(::Nothing, i, j, nx, ny, u, v, hamada) = zero(u)

@inline function _urban_speed(urban, i, j, nx, ny, u, v, hamada)
    @inbounds plan = urban.plan[i, j, 1]
    @inbounds separation = urban.separation[i, j, 1]
    @inbounds nonburnable = urban.nonburnable[i, j, 1]
    U = sqrt(u^2 + v^2)
    wx, wy = u / _pos(U), v / _pos(U)      # zero vector when calm; all rates equal then
    head, flank, back = hamada_rates(hamada, U, plan, separation, nonburnable)
    speed = ellipse_speed(nx, ny, wx, wy, head, flank, back)
    return ifelse(plan > 0, speed, zero(speed))
end

@kernel function _spread_rate!(speed, φ, grid, fuel, moisture, shape, u, v, ∂x_h, ∂y_h, urban, hamada, t)
    i, j = @index(Global, NTuple)
    # Central differences of φ at the cell centre; `_pos` makes the normal zero
    # rather than NaN where ∇φ = 0.
    φx = ℑxᶜᵃᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, φ)
    φy = ℑyᵃᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, φ)
    m = _pos(sqrt(φx^2 + φy^2))
    nx, ny = φx / m, φy / m
    uᵢ, vᵢ = _at(u, i, j, t), _at(v, i, j, t)
    hx, hy = _at(∂x_h, i, j, t), _at(∂y_h, i, j, t)
    vegetation = _vegetation_speed(shape, _bed(fuel, i, j), moisture, nx, ny, uᵢ, vᵢ, hx, hy)
    built = _urban_speed(urban, i, j, nx, ny, uᵢ, vᵢ, hamada)
    @inbounds speed[i, j, 1] = max(vegetation, built)
end
