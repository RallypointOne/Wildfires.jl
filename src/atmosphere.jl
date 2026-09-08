using Oceananigans.Grids: znode
using Oceananigans.OutputReaders: FieldTimeSeries

#-----------------------------------------------------------------------------# raster_column_series
"""
    raster_column_series(values, heights, times, grid, crs::ProjectedCRS;
                         method = :bilinear, time_indexing = Clamp()) -> FieldTimeSeries

A `FieldTimeSeries` at cell centres of the three-dimensional `grid` from
level data such as an NWP analysis on pressure levels: `values[n]` and
`heights[n]` are multi-band rasters for time `times[n]`, band `l` holding the
variable and the geometric height [m, same datum as the grid's `z`] of level
`l`. Each band is sampled through [`raster_sampler`](@ref) at every column,
then interpolated linearly in height onto the column's cell centres, holding
the lowest and highest levels constant beyond their range. Column work runs
on the CPU at setup.

Requires Rasters.jl and ArchGDAL.jl to be loaded.

### Examples
```julia
u = raster_column_series([Raster("hrrr_UGRD_\$t.tif") for t in stamps],
                         [Raster("hrrr_HGT_\$t.tif") for t in stamps], times, grid, crs)
set!(model; u = interior(u[1]))
```
"""
function raster_column_series(values, heights, times, grid, crs::ProjectedCRS;
                              method = :bilinear, time_indexing = Clamp())
    length(values) == length(heights) == length(times) ||
        throw(ArgumentError("need one values raster, one heights raster, and one time per snapshot"))
    Nx, Ny, Nz = size(grid)
    xc, yc = xnodes(grid, Center()), ynodes(grid, Center())
    series = FieldTimeSeries{Center, Center, Center}(grid, times; time_indexing)
    A = zeros(Float64, Nx, Ny, Nz)

    for (n, (value, height)) in enumerate(zip(values, heights))
        L = size(value, 3)
        size(height, 3) == L || throw(ArgumentError("values and heights differ in level count at snapshot $n"))
        v = [raster_sampler(value[:, :, l], crs; method) for l in 1:L]
        h = [raster_sampler(height[:, :, l], crs; method) for l in 1:L]
        Threads.@threads for j in 1:Ny
            zs = zeros(L); vs = zeros(L)
            for i in 1:Nx
                for l in 1:L
                    zs[l] = h[l](xc[i], yc[j])
                    vs[l] = v[l](xc[i], yc[j])
                end
                order = sortperm(zs)
                for k in 1:Nz
                    A[i, j, k] = _interpolate_column(zs, vs, order, znode(i, j, k, grid, Center(), Center(), Center()))
                end
            end
        end
        set!(series[n], eltype(grid).(A))
    end
    return series
end

# Linear interpolation of (zs, vs) at z, using `order` to walk the levels by
# increasing height; constant beyond the ends.
function _interpolate_column(zs, vs, order, z)
    lo = order[1]
    z <= zs[lo] && return vs[lo]
    for m in 2:length(order)
        hi = order[m]
        if z <= zs[hi]
            w = (z - zs[lo]) / max(zs[hi] - zs[lo], eps())
            return (1 - w) * vs[lo] + w * vs[hi]
        end
        lo = hi
    end
    return vs[lo]
end

#-----------------------------------------------------------------------------# lateral_sponge
"""
    lateral_sponge(grid; width = 5, ramp = :cosine) -> Field

Horizontal mask for lateral relaxation toward a parent state (a Davies
sponge): 1 at the four side walls, falling to 0 over `width` cells into the
interior along a half-cosine (`ramp = :cosine`) or linearly (`:linear`), the
maximum over the four sides. A `Field{Center, Center, Nothing}` on the
grid's architecture, so it indexes as `mask[i, j, 1]` inside a kernel.

### Examples
```julia
mask = lateral_sponge(grid; width = 8)
```
"""
function lateral_sponge(grid; width::Integer = 5, ramp = :cosine)
    width >= 1 || throw(ArgumentError("width must be at least one cell"))
    ramp in (:cosine, :linear) || throw(ArgumentError("ramp must be :cosine or :linear, got $ramp"))
    Nx, Ny = size(grid, 1), size(grid, 2)
    shape(d) = d >= width ? 0.0 : (ramp === :linear ? 1 - d / width : (1 + cos(π * d / width)) / 2)
    mask = [max(shape(i - 0.5), shape(Nx - i + 0.5), shape(j - 0.5), shape(Ny - j + 0.5))
            for i in 1:Nx, j in 1:Ny]
    return _horizontal_field(grid, mask)
end

#-----------------------------------------------------------------------------# relaxation_forcing
"""
    relaxation_forcing(name::Symbol, target, rate; mask = 1)

An Oceananigans `Forcing` that relaxes the model field `name` (`:u`, `:v`,
`:θ`, `:qᵛ`, ...) toward `target` at `rate` [1/s] times `mask`:
`rate * mask * (target - field)`. `target` is a number, a field, or a
`FieldTimeSeries` on the model grid, read at the model clock; `mask` is a
number or a horizontal field such as [`lateral_sponge`](@ref). Pass it to
Breeze under the same specific key, `forcing = (u = relaxation_forcing(:u, …), …)`,
where Breeze multiplies by density. Face-located velocities read the target
at the cell centre, an offset of half a cell that is immaterial in a sponge.

### Examples
```julia
mask = lateral_sponge(grid; width = 8)
forcing = (u = relaxation_forcing(:u, u_series, 1/600; mask),
           v = relaxation_forcing(:v, v_series, 1/600; mask),
           θ = relaxation_forcing(:θ, θ_series, 1/600; mask))
model = AtmosphereModel(grid; dynamics, forcing)
```
"""
function relaxation_forcing(name::Symbol, target, rate; mask = 1)
    parameters = (; name = Val(name), target, rate, mask)
    return Forcing(_relax; discrete_form = true, parameters)
end

@inline _field_of(fields, ::Val{N}) where {N} = getproperty(fields, N)

@inline function _relax(i, j, k, grid, clock, fields, p)
    target = _at(p.target, i, j, k, clock.time)
    @inbounds return p.rate * _at(p.mask, i, j) * (target - _field_of(fields, p.name)[i, j, k])
end

@inline _at(w::Number, i, j, k, t) = w
@inline _at(w::FlavorOfFTS, i, j, k, t) = @inbounds w[i, j, k, Time(t)]
@inline _at(w, i, j, k, t) = @inbounds w[i, j, k]
