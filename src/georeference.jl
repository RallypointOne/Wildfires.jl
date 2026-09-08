using Proj
import GeoFormatTypes as GFT

#-----------------------------------------------------------------------------# UTM utilities
"""
    utm_zone(lon) -> Int

UTM zone number (1–60) for longitude `lon` in degrees.

### Examples
```julia
utm_zone(-105.2)  # 13
```
"""
utm_zone(lon) = clamp(floor(Int, (lon + 180) / 6) + 1, 1, 60)

"""
    utm_epsg(lon, lat) -> Int

EPSG code of the UTM zone containing `(lon, lat)` in degrees: `326xx` in the
northern hemisphere, `327xx` in the southern.

### Examples
```julia
utm_epsg(-105.2, 40.0)  # 32613
utm_epsg(151.2, -33.9)  # 32756
```
"""
utm_epsg(lon, lat) = (lat >= 0 ? 32600 : 32700) + utm_zone(lon)

#-----------------------------------------------------------------------------# ProjectedCRS
"""
    ProjectedCRS(epsg, origin)
    ProjectedCRS(lon, lat; epsg = utm_epsg(lon, lat))

Georeferencing for an Oceananigans grid whose `x`/`y` are in meters.

`epsg` names the projected CRS the grid lives in; `origin` is that CRS's
coordinate of the grid's `(0, 0)`, so grid meters and projected meters differ
by a translation. The two-argument form projects `(lon, lat)` into `epsg` and
uses the result as the origin, placing that point at the grid's `(0, 0)`.

The grid itself carries no CRS — pass a `ProjectedCRS` alongside it wherever
real-world coordinates are involved.

### Examples
```julia
crs = ProjectedCRS(-105.18, 39.96)   # UTM 13N, origin at the Marshall Fire
to_grid(crs, -105.18, 39.96)         # ≈ (0.0, 0.0)
```
"""
struct ProjectedCRS
    epsg::Int
    origin::NTuple{2, Float64}
    # Explicit inner constructor: the auto-generated untyped one would collide
    # with the `(lon, lat)` method below.
    ProjectedCRS(epsg::Integer, origin::NTuple{2, Real}) = new(Int(epsg), Float64.(origin))
end

function ProjectedCRS(lon::Real, lat::Real; epsg = utm_epsg(lon, lat))
    transform = Proj.Transformation("EPSG:4326", "EPSG:$epsg"; always_xy = true)
    return ProjectedCRS(epsg, transform(lon, lat))
end

Base.show(io::IO, crs::ProjectedCRS) =
    print(io, "ProjectedCRS(EPSG:", crs.epsg, ", origin=", crs.origin, ")")

#-----------------------------------------------------------------------------# Transformations
# Proj takes CRS strings; rasters carry GeoFormatTypes wrappers (WellKnownText,
# EPSG, ProjString, ...). Unwrap to the string Proj expects.
_crs_string(source::AbstractString) = String(source)
_crs_string(source::GFT.GeoFormat) = String(GFT.val(source))
_crs_string(source::GFT.EPSG) = "EPSG:" * join(GFT.val(source), '+')

"""
    to_grid_transform(crs::ProjectedCRS, source) -> Function

Return `(X, Y) -> (x, y)` mapping coordinates in `source` — anything Proj
accepts, e.g. `"EPSG:4326"` or `"EPSG:5070"` — to grid-local meters.

Build the transform once and reuse it when converting many points; constructing
a `Proj.Transformation` is the expensive part.

### Examples
```julia
f = to_grid_transform(crs, "EPSG:4326")
f(-105.18, 39.96)
```
"""
function to_grid_transform(crs::ProjectedCRS, source)
    transform = Proj.Transformation(_crs_string(source), "EPSG:$(crs.epsg)"; always_xy = true)
    x₀, y₀ = crs.origin
    return function (X, Y)
        x, y = transform(X, Y)
        return (x - x₀, y - y₀)
    end
end

"""
    from_grid_transform(crs::ProjectedCRS, target) -> Function

Return `(x, y) -> (X, Y)` mapping grid-local meters to coordinates in `target`.
The inverse of [`to_grid_transform`](@ref).

### Examples
```julia
f = from_grid_transform(crs, "EPSG:4326")
f(0.0, 0.0)  # the lon/lat of the grid origin
```
"""
function from_grid_transform(crs::ProjectedCRS, target)
    transform = Proj.Transformation("EPSG:$(crs.epsg)", _crs_string(target); always_xy = true)
    x₀, y₀ = crs.origin
    return function (x, y)
        return transform(x + x₀, y + y₀)
    end
end

"""
    to_grid(crs::ProjectedCRS, lon, lat) -> (x, y)

Convert WGS84 `(lon, lat)` in degrees to grid-local meters. Builds a
transformation per call — use [`to_grid_transform`](@ref) for many points.
"""
to_grid(crs::ProjectedCRS, lon, lat) = to_grid_transform(crs, "EPSG:4326")(lon, lat)

"""
    to_lonlat(crs::ProjectedCRS, x, y) -> (lon, lat)

Convert grid-local meters to WGS84 `(lon, lat)` in degrees. Builds a
transformation per call — use [`from_grid_transform`](@ref) for many points.
"""
to_lonlat(crs::ProjectedCRS, x, y) = from_grid_transform(crs, "EPSG:4326")(x, y)

#-----------------------------------------------------------------------------# Raster ingest
"""
    raster_sampler(raster, crs::ProjectedCRS; method = :bilinear, fill_value = 0) -> Function

Return `(x, y) -> value`, sampling `raster` at grid-local meters `(x, y)`, for
use with `set!` on a field or as the `topography` argument of Breeze's
`materialize_terrain!`.

`raster` is reprojected once into `crs` with `Rasters.resample` (GDAL warp), so
it may be in any CRS GDAL understands; each query then looks up the
reprojected raster. `method` is `:bilinear` for continuous data (elevation,
wind) or `:near` for categorical data (fuel model codes, aspect), and applies
to both the warp and the lookup. Points off the raster get `fill_value`.

Requires Rasters.jl and ArchGDAL.jl to be loaded.

### Examples
```julia
fuel_code = raster_sampler(Raster("fuel.tif"), crs; method = :near)
set!(field, fuel_code)
```
"""
function raster_sampler end

"""
    raster_topography(raster, crs::ProjectedCRS; fill_value = 0) -> Function

Return `(x, y) -> h`, the surface elevation at grid-local meters `(x, y)`, for
use as the `topography` argument of Breeze's `materialize_terrain!`. Equivalent
to [`raster_sampler`](@ref) with `method = :bilinear`.

### Examples
```julia
h = raster_topography(Raster("elevation.tif"), crs)
materialize_terrain!(grid, h)
```
"""
raster_topography(raster, crs::ProjectedCRS; fill_value = 0) =
    raster_sampler(raster, crs; method = :bilinear, fill_value)
