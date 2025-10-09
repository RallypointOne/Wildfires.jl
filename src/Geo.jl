module Geo

import GeoInterface as GI
import GeometryOps as GO
import Proj
import GMT
import Extents
import H3

using StyledStrings

#-----------------------------------------------------------------------------# LatLon
# Wrapper for latitude/longitude coordinates in degrees
struct LatLon{T}
    lat::T
    lon::T
end
LatLon(o::H3.API.LatLng) = LatLon(rad2deg(o.lat), rad2deg(o.lng))
Base.show(io::IO, o::LatLon) = print(io, (o.lat, o.lon))

#-----------------------------------------------------------------------------# H3Cell
struct H3Cell{T}
    index::UInt64
    data::T
    function H3Cell(index::UInt64, data::T = nothing) where {T}
        H3.API.isValidCell(index) || throw(ArgumentError("Invalid H3 index: $index"))
        new{T}(index, data)
    end
end
H3Cell(o::LatLon, data=nothing; resolution=15) = H3Cell(H3.API.latLngToCell(H3.API.LatLng(deg2rad(o.lat), deg2rad(o.lon)), resolution), data)
H3Cell(x::AbstractString, data=nothing; resolution=15) = H3Cell(geocode(x), data; resolution)

resolution(o::H3Cell) = H3.API.getResolution(o.index)
boundary(o::H3Cell) = LatLon.(H3.API.cellToBoundary(o.index))
center(o::H3Cell) = LatLon(H3.API.cellToLatLng(o.index))
is_pentagon(o::H3Cell) = H3.API.isPentagon(o.index)
area(o::H3Cell) = H3.API.cellAreaM2(o.index)  # in m²

grid_disk(o::H3Cell, k::Integer) = H3.API.gridDisk(o.index, k)
neighbors(o::H3Cell) = filter(!=(o.index), grid_disk(o, 1))
function rings(o::H3Cell, k::Integer)
    k > 0 || throw(ArgumentError("k must be positive"))
    out = [grid_disk(o, 1)]
    for i in 2:k
        push!(out, setdiff!(grid_disk(o, i), out[i-1]))
    end
    filter!(!=(o.index), out[1])
    return out
end

function Base.show(io::IO, o::T) where {T <: H3Cell}
    shape = is_pentagon(o) ? styled"{bright_red:⬠}" : styled"{bright_green:⬡}"
    res = resolution(o)
    print(io, styled"{bright_cyan:$T} $shape {bright_magenta:$res} {bright_black:$(center(o))}")
end


# TODO: plotting, convert polygon to H3 cells

#-----------------------------------------------------------------------------# geocode
function geocode(x::AbstractString)
    gmt = GMT.geocoder(x)
    LatLon(Proj.Transformation(gmt.proj4, "WGS84")(gmt.data...)...)
end


#-----------------------------------------------------------------------------# guess_zoom
"""
    guess_zoom(ext; crs=:wgs84, viewport=(1024, 768), tile=256, minzoom=0, maxzoom=22)

Return an integer Web-Mercator zoom level that will fit the given `ext` into the
pixel `viewport` (width,height). `ext` may be an `Extents.Extent` (with fields
`X=(xmin,xmax), Y=(ymin,ymax)`) or a 4-tuple `(xmin, ymin, xmax, ymax)`.

`crs` can be `:wgs84` (lon/lat degrees) or `:webmercator` (meters).
"""
function guess_zoom(ext::Extents.Extent; crs=:wgs84, viewport=(1024, 768), tile=256, minzoom=0, maxzoom=22)
    # --- constants for Web Mercator ---
    R = 6378137.0
    WORLD_M = 2π * R                # ~40,075,016.685 m
    tilepixels(z) = tile * 2.0^z
    # --- extract bounds ---
    xmin, ymin, xmax, ymax = (ext.X[1], ext.Y[1], ext.X[2], ext.Y[2])
    # --- project to Web Mercator meters if needed ---
    if crs == :wgs84
        clamplat(φ) = max(-85.05112878, min(85.05112878, φ))
        lon2x(λ) = R * deg2rad(λ)
        lat2y(φ) = R * log(tan(π/4 + deg2rad(clamplat(φ))/2))

        # Handle anti-meridian simply by unwrapping longitudes if needed
        λmin, λmax = xmin, xmax
        if λmax < λmin
            λmax += 360
        end
        x0, x1 = lon2x(λmin), lon2x(λmax)
        y0, y1 = lat2y(ymin), lat2y(ymax)
    elseif crs == :webmercator
        x0, y0, x1, y1 = xmin, ymin, xmax, ymax
    else
        error("Unsupported crs: $crs (use :wgs84 or :webmercator)")
    end

    w = abs(x1 - x0)
    h = abs(y1 - y0)
    vw, vh = viewport
    if w <= 0 || h <= 0 || vw <= 0 || vh <= 0
        return minzoom
    end

    # meters per pixel required to fit bbox
    req_m_per_px = max(w / vw, h / vh)

    # meters per pixel at zoom z: WORLD_M / tilepixels(z)
    # Find largest z such that mpp(z) <= req_m_per_px  ->  2^z >= WORLD_M / (tile * req_m_per_px)
    z = floor(Int, log2(WORLD_M / (tile * req_m_per_px)))

    clamp(z, minzoom, maxzoom)
end


end
