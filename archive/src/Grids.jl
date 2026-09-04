module Grids

using Extents: Extent
using GeoFormatTypes: CoordinateReferenceSystemFormat, EPSG

export Grid, utm_zone, utm_epsg, xcoords, ycoords

#-----------------------------------------------------------------------------# Grid
mutable struct Grid{C, T, S, L <: NamedTuple}
    crs::C
    extent::Extent{(:X, :Y), Tuple{Tuple{T, T}, Tuple{T, T}}}
    dx::T  # grid spacing (m)
    t::T   # timestamp (s)
    state::Matrix{S}
    layers::L  # Layers that match dims of state
end

#-----------------------------------------------------------------------------# Lon/Lat to Local Meters
const WGS84_A = 6378137.0          # semi-major axis [m]
const WGS84_E2 = 0.00669437999014  # eccentricity squared

"""
    _lonlat_to_meters(lon, lat, lon0, lat0) -> (x, y)

Convert (lon, lat) to local (x, y) in meters relative to (lon0, lat0).
Uses the WGS84 ellipsoid radii of curvature for an accurate flat-Earth
approximation (sub-meter error within ~50 km of the origin).
"""
function _lonlat_to_meters(lon, lat, lon0, lat0)
    φ = deg2rad(lat0)
    sinφ = sin(φ)
    N = WGS84_A / sqrt(1 - WGS84_E2 * sinφ^2)             # prime vertical radius
    M = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * sinφ^2)^1.5  # meridional radius
    x = N * cos(φ) * deg2rad(lon - lon0)
    y = M * deg2rad(lat - lat0)
    return (x, y)
end

"""
    Grid(lon, lat, dx; distance=1000.0, north=distance, south=distance, east=distance, west=distance)

Create a `Grid` centered at `(lon, lat)` in degrees with cell spacing `dx` [m].

The grid extends `north`, `south`, `east`, and `west` meters from the center.
If `distance` is provided, it sets all four directions to the same value.

The CRS is automatically set to the appropriate UTM zone.

### Examples
```julia
# 2km × 2km grid centered on Boulder, CO with 30m cells
grid = Grid(-105.2, 40.0, 30.0; distance=1000.0)

# Asymmetric: 5km north, 1km south
grid = Grid(-105.2, 40.0, 30.0; north=5000.0, south=1000.0, east=3000.0, west=3000.0)
```
"""
function Grid(lon, lat, dx; distance=1000.0,
              north=distance, south=distance, east=distance, west=distance)
    crs = utm_epsg(lon, lat)
    cx, cy = _lonlat_to_meters(lon, lat, lon, lat)  # (0, 0) at center
    x_lo = cx - west
    x_hi = cx + east
    y_lo = cy - south
    y_hi = cy + north
    # Snap to grid spacing
    nx = round(Int, (x_hi - x_lo) / dx)
    ny = round(Int, (y_hi - y_lo) / dx)
    nx = max(nx, 1)
    ny = max(ny, 1)
    x_hi = x_lo + nx * dx
    y_hi = y_lo + ny * dx
    extent = Extent(X=(x_lo, x_hi), Y=(y_lo, y_hi))
    state = zeros(ny, nx)
    Grid(crs, extent, Float64(dx), 0.0, state, NamedTuple())
end

#-----------------------------------------------------------------------------# UTM Utilities
"""
    utm_zone(lon) -> Int

UTM zone number (1–60) for a given longitude in degrees.

### Examples
```julia
utm_zone(-105.2)  # 13 (Boulder, CO)
```
"""
utm_zone(lon) = clamp(floor(Int, (lon + 180) / 6) + 1, 1, 60)

"""
    utm_epsg(lon, lat) -> EPSG

EPSG code for the UTM zone containing the given longitude/latitude (in degrees).
Returns `EPSG(326xx)` for the northern hemisphere and `EPSG(327xx)` for the southern.

### Examples
```julia
utm_epsg(-105.2, 40.0)  # EPSG(32613) — Boulder, CO
utm_epsg(151.2, -33.9)  # EPSG(32756) — Sydney
```
"""
utm_epsg(lon, lat) = EPSG((lat >= 0 ? 32600 : 32700) + utm_zone(lon))

#-----------------------------------------------------------------------------# Size + Coordinates
Base.size(g::Grid) = size(g.state)

"""
    xcoords(g::Grid)

Return the x-coordinates of grid cell centers.
"""
function xcoords(g::Grid)
    x_lo = g.extent.X[1]
    range(x_lo + g.dx / 2, step=g.dx, length=size(g.state, 2))
end

"""
    ycoords(g::Grid)

Return the y-coordinates of grid cell centers.
"""
function ycoords(g::Grid)
    y_lo = g.extent.Y[1]
    range(y_lo + g.dx / 2, step=g.dx, length=size(g.state, 1))
end

end # module
