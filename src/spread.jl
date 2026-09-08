using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ

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

#-----------------------------------------------------------------------------# spread_rate!
"""
    spread_rate!(speed, model::FireModel, bed::FuelBed, moisture::FuelClasses;
                 wind = (0, 0), slope = (0, 0), structures = nothing,
                 hamada = HamadaModel())

Fill `speed`, a field on `model.grid`, with the rate of spread [m/s] normal
to the fire front in every cell, for use as the `speed` argument of
[`advance!`](@ref).

`wind` is the open 20 ft wind `(u, v)` [m/s] and `slope` the terrain gradient
`(∂h/∂x, ∂h/∂y)` as rise over run; each component is a number or a field on
`model.grid`. Vegetation spreads at the Rothermel rate of `bed` with the wind
reduced to midflame height by [`wind_adjustment`](@ref): as in WRF-SFIRE,
wind and slope are resolved onto the outward normal `n = ∇φ/|∇φ|`, and
[`spread_rate`](@ref) treats a negative component as calm or flat, so the
front spreads at the head-fire rate downwind and upslope and at the no-wind,
no-slope rate against them. Where `∇φ` vanishes the normal is undefined and
the no-wind rate is used; that happens only away from the front.

With `structures`, cells within their neighbourhood also spread at the Hamada
urban rate — [`hamada_rates`](@ref) from the open wind and the structure
fields, shaped into an ellipse by [`ellipse_speed`](@ref) — and the cell
takes the larger of the two rates.

`bed` and `moisture` apply to every cell. Recompute after each
[`advance!`](@ref), since the normal changes as the front moves.

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
function spread_rate!(speed, model::FireModel, bed::FuelBed, moisture::FuelClasses;
                      wind = (0, 0), slope = (0, 0), structures = nothing,
                      hamada = HamadaModel())
    grid = model.grid
    FT = eltype(grid)
    u, v = _component(wind[1], FT), _component(wind[2], FT)
    ∂x_h, ∂y_h = _component(slope[1], FT), _component(slope[2], FT)
    bed = FuelBed{FT}(bed)
    urban = _urban_fields(structures)

    launch!(architecture(grid), grid, :xy, _spread_rate!, speed, model.φ, grid,
            bed, FuelClasses{FT}(moisture), wind_adjustment(bed), u, v, ∂x_h, ∂y_h,
            urban, HamadaModel{FT}(hamada))
    fill_halo_regions!(speed)

    return speed
end

_component(c::Number, FT) = FT(c)
_component(c, FT) = c

_urban_fields(::Nothing) = nothing
_urban_fields(s::Structures) = (; plan = s.plan_dimension, separation = s.separation,
                                  nonburnable = s.nonburnable_fraction)

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

@kernel function _spread_rate!(speed, φ, grid, bed, moisture, waf, u, v, ∂x_h, ∂y_h, urban, hamada)
    i, j = @index(Global, NTuple)
    # Central differences of φ at the cell centre; `_pos` makes the normal zero
    # rather than NaN where ∇φ = 0.
    φx = ℑxᶜᵃᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, φ)
    φy = ℑyᵃᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, φ)
    m = _pos(sqrt(φx^2 + φy^2))
    nx, ny = φx / m, φy / m
    uᵢ, vᵢ = _at(u, i, j), _at(v, i, j)
    wind_n = uᵢ * nx + vᵢ * ny
    slope_n = _at(∂x_h, i, j) * nx + _at(∂y_h, i, j) * ny
    vegetation = spread_rate(bed, moisture, waf * wind_n, slope_n)
    built = _urban_speed(urban, i, j, nx, ny, uᵢ, vᵢ, hamada)
    @inbounds speed[i, j, 1] = max(vegetation, built)
end
