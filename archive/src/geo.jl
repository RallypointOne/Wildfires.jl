#-----------------------------------------------------------------------------# LatLon
"""
    LatLon(lat::T, lon::T) where {T<:Real}

A simple struct to represent a latitude/longitude pair in *degrees* (EPSG:4326).
"""
struct LatLon{T}
    lat::T
    lon::T
end
LatLon(o::API.LatLng) = LatLon(rad2deg(o.lat), rad2deg(o.lng))
LatLon(; lat, lon) = LatLon(lat, lon)
LatLon(x::AbstractVector{<:Real}) = length(x) == 2 ? LatLon(x[2], x[1]) : throw(ArgumentError("Expected 2-element vector, got length $(length(x))"))
LatLon(x::Tuple{<:Real, <:Real}) = LatLon(x[2], x[1])

Base.NamedTuple((; lon, lat)::LatLon) = (x=lon, y=lat)
Base.show(io::IO, o::LatLon) = print(io, NamedTuple(o))

H3.Lib.LatLng(o::LatLon) = H3.Lib.LatLng(deg2rad(o.lat), deg2rad(o.lon))

GI.isgeometry(::LatLon) = true
GI.geomtrait(::LatLon) = GI.PointTrait()
GI.coordinates(::GI.PointTrait, o::LatLon) = (o.lon, o.lat)
GI.getcoord(::GI.PointTrait, o::LatLon, i::Integer) = GI.coordinates(o)[i]
GI.ncoord(::GI.PointTrait, ::LatLon) = 2
GI.@enable_makie Makie LatLon

"Approximate radius of the earth in meters (WGS84)."
const R = 6_371_000

"""
    haversine(a::LatLon, b::LatLon)
Calculate the great-circle distance (meters) between two LatLon points using the Haversine formula.
"""
function haversine(a::LatLon{T}, b::LatLon{T}) where {T <: Real}
    x = sind((b.lat - a.lat) / 2) ^ 2 + cosd(a.lat) * cosd(b.lat) * sind((b.lon - a.lon) / 2)^2
    return 2R * asin(min(sqrt(x), one(x)))
end

"""
    destination(a::LatLon, bearing°, dist_m)

Find destination point given starting point (LatLon), bearing (clockwise from North), and distance (m)
"""
function destination(a::LatLon, bearing°, dist)
    ϕ1, λ1 = a.lat, a.lon
    δ = rad2deg(dist / R)
    ϕ2 = asind(sind(ϕ1) * cosd(δ) + cosd(ϕ1) * sind(δ) * cosd(bearing°))
    λ2 = λ1 + atand(sind(bearing°) * sind(δ) * cosd(ϕ1), cosd(δ) - sind(ϕ1) * sind(ϕ2))
    LatLon(ϕ2, λ2)
end

#-----------------------------------------------------------------------------# extent
function Extents.extent(o::LatLon; n=1000, s=1000, e=1000, w=1000)
    N = destination(o, 0, n).lat
    E = destination(o, 90, e).lon
    S = destination(o, 180, s).lat
    W = destination(o, 270, w).lon
    return Extents.Extent(X=(W, E), Y=(S, N))
end
Extents.extent(o::LatLon, meters) = Extents.extent(o; n=meters, s=meters, e=meters, w=meters)

#-----------------------------------------------------------------------------# geocode
function geocode(x::AbstractString)
    gmt = GMT.geocoder(x)
    LatLon(Proj.Transformation(gmt.proj4, "WGS84")(gmt.data...)...)
end


#-----------------------------------------------------------------------------# H3.API wrappers
resolution(index::UInt64) = API.getResolution(index)

is_cell(index::UInt64) = API.isValidCell(index)
is_directed_edge(index::UInt64) = API.isValidDirectedEdge(index)
is_vertex(index::UInt64) = API.isValidVertex(index)
is_pentagon(index::UInt64) = API.isPentagon(index)

check(f, x::UInt64) = f(x) ? x : throw(ArgumentError("Invalid H3 index: $x does not satisfy condition $f"))


#-----------------------------------------------------------------------------# H3IndexType
abstract type H3IndexType end

GI.isgeometry(::H3IndexType) = true
GI.geomtrait(::H3IndexType) = GI.UnknownTrait()
GI.crs(::H3IndexType) = GFT.EPSG(4326)
GI.ncoord(::GI.PolygonTrait, o::H3IndexType) = 2
GI.@enable_makie Makie H3IndexType

is_cell(o::H3IndexType) = is_cell(o.index)
is_directed_edge(o::H3IndexType) = is_directed_edge(o.index)
is_vertex(o::H3IndexType) = is_vertex(o.index)
resolution(o::H3IndexType) = resolution(o.index)
is_pentagon(o::H3IndexType) = is_pentagon(o.index)

#-----------------------------------------------------------------------------# Cell
struct Cell <: H3IndexType
    index::UInt64
    Cell(x::UInt64) = new(check(is_cell, x))
end
Cell(o::LatLon, resolution=10) = Cell(API.latLngToCell(API.LatLng(deg2rad(o.lat), deg2rad(o.lon)), resolution))
Cell(x::AbstractString) = Cell(geocode(x))

GI.geomtrait(::Cell) = GI.PolygonTrait()
GI.centroid(::GI.PolygonTrait, o::Cell) = LatLon(API.cellToLatLng(o.index))
GI.coordinates(::GI.PolygonTrait, o::Cell) = (out = LatLon.(API.cellToBoundary(o.index)); return [out..., out[1]])
GI.nhole(::GI.PolygonTrait, o::Cell) = 0
GI.ngeom(::GI.PolygonTrait, o::Cell) = 1
GI.getgeom(::GI.PolygonTrait, o::Cell, i::Integer) = GI.LineString(GI.coordinates(o))
GI.area(::GI.PolygonTrait, o::Cell) = area_m2(o)  # in m²

function Base.show(io::IO, o::Cell)
    shape = is_pentagon(o) ? styled"{bright_red:⬠}" : styled"{bright_green:⬡}"
    ll = styled"{bright_black:$(GI.centroid(o))}"
    print(io, styled"$shape {bright_cyan:$(typeof(o))} {bright_magenta:$(resolution(o))} {bright_black:$(repr(o.index))} $ll")
end

vertices(o::Cell) = Vertex.(API.cellToVertexes(o.index))
const vertexes = vertices

area_m2(o::Cell) = API.cellAreaM2(o.index)  # in m²

grid_path_cells(a::Cell, b::Cell) = Cell.(API.gridPathCells(a.index, b.index))

grid_disk(o::Cell, k::Integer) = Cell.(API.gridDisk(o.index, k))
grid_ring_unsafe(o::Cell, k::Integer) = Cell.(API.gridRingUnsafe(o.index, k))

Base.getindex(o::Cell, i::Integer) = Vertex(API.cellToVertex(o.index, i))
Base.getindex(o::Cell, i::Integer, j::Integer) = GridIJ(o)[i, j]


#-----------------------------------------------------------------------------# cells (geom-to-Vector{Cell})
cells(geom, res::Integer = 10) = cells(GI.trait(geom), geom, res)

cells(trait::GI.PointTrait, geom, res::Integer) = [Cell(LatLon(GI.coordinates(trait, geom)...), res)]

function cells(trait::GI.MultiPointTrait, geom, res::Integer)
    unique!(Cell.(LatLon.(GI.coordinates(trait, geom), res)))
end

function cells(trait::GI.LineStringTrait, geom, res::Integer)
    coords = GI.coordinates(trait, geom)
    out = [Cell(LatLon(reverse(coords[1])...), res)]
    for (lon, lat) in @view coords[2:end]
        c = Cell(LatLon(lat, lon), res)
        path = grid_path_cells(out[end], c)
        append!(out, path[2:end])
    end
    unique!(out)
end

function cells(trait::GI.PolygonTrait, geom, res::Integer)
    verts = [H3.Lib.LatLng.(LatLon.(ring)) for ring in GI.coordinates(trait, geom)]
    GC.@preserve verts begin
        geo_loops = H3.Lib.GeoLoop.(length.(verts), pointer.(verts))
    end
    GC.@preserve geo_loops begin
        n = length(geo_loops)
        geo_polygon = H3.Lib.GeoPolygon(geo_loops[1], n - 1, n > 1 ? pointer(geo_loops, 2) : C_NULL)
    end
    GC.@preserve geo_polygon begin
        max_n = Ref{Int64}()
        ret::API.H3Error = H3.Lib.maxPolygonToCellsSize(Ref(geo_polygon), res, 0, max_n)
        API._check_h3error(ret, nothing)
        out = zeros(UInt64, max_n[])
        ret2::API.H3Error = H3.Lib.polygonToCells(Ref(geo_polygon), res, 0, out)
        API._check_h3error(ret2, out)
        out = Cell.(filter!(!iszero, unique!(out)))
    end
    # Hack around the fact that the libh3 implementation sometimes misses cells on the edges.
    for ring in GI.coordinates(geom)
        union!(out, cells(GI.LineString(ring), res))
    end
    return out
end

function cells(trait::GI.MultiPolygonTrait, geom, res::Integer)
    reduce(union, cells.(GI.getpolygon(geom), res))
end

function cells(trait::GI.RasterTrait, geom, res::Integer)
    # TODO
end

#-----------------------------------------------------------------------------# GridIJ
struct GridIJ
    origin::Cell
    ij::API.CoordIJ
end
GridIJ(o::Cell) = GridIJ(o, API.cellToLocalIj(o.index, o.index))
Base.show(io::IO, o::GridIJ) = print(io, styled"GridIJ - origin: $(o.origin)")
function Base.getindex(grid::GridIJ, i::Integer, j::Integer)
    Cell(API.localIjToCell(grid.origin.index, API.CoordIJ(i + grid.ij.i, j + grid.ij.j)))
end


#-----------------------------------------------------------------------------# Vertex
struct Vertex <: H3IndexType
    index::UInt64
    Vertex(x::UInt64) = new(check(is_vertex, x))
end
GI.geomtrait(::Vertex) = GI.PointTrait()
GI.coordinates(::GI.PointTrait, o::Vertex) = LatLon(API.vertexToLatLng(o.index))
GI.getcoord(::GI.PointTrait, o::Vertex, i::Integer) = GI.coordinates(o)[i]

function Base.show(io::IO, o::Vertex)
    ll = styled"{bright_black:$(GI.coordinates(o))}"
    print(io, styled"{bright_cyan:$(typeof(o))} {bright_magenta:$(resolution(o))} {bright_black:$(repr(o.index))} $ll")
end

LatLon(o::Vertex) = LatLon(API.vertexToLatLng(o.index))

# #-----------------------------------------------------------------------------# guess_zoom
# """
#     guess_zoom(ext; crs=:wgs84, viewport=(1024, 768), tile=256, minzoom=0, maxzoom=22)

# Return an integer Web-Mercator zoom level that will fit the given `ext` into the
# pixel `viewport` (width,height). `ext` may be an `Extents.Extent` (with fields
# `X=(xmin,xmax), Y=(ymin,ymax)`) or a 4-tuple `(xmin, ymin, xmax, ymax)`.

# `crs` can be `:wgs84` (lon/lat degrees) or `:webmercator` (meters).
# """
# function guess_zoom(ext::Extents.Extent; crs=:wgs84, viewport=(1024, 768), tile=256, minzoom=0, maxzoom=22)
#     # --- constants for Web Mercator ---
#     R = 6378137.0
#     WORLD_M = 2π * R                # ~40,075,016.685 m
#     tilepixels(z) = tile * 2.0^z
#     # --- extract bounds ---
#     xmin, ymin, xmax, ymax = (ext.X[1], ext.Y[1], ext.X[2], ext.Y[2])
#     # --- project to Web Mercator meters if needed ---
#     if crs == :wgs84
#         clamplat(φ) = max(-85.05112878, min(85.05112878, φ))
#         lon2x(λ) = R * deg2rad(λ)
#         lat2y(φ) = R * log(tan(π/4 + deg2rad(clamplat(φ))/2))

#         # Handle anti-meridian simply by unwrapping longitudes if needed
#         λmin, λmax = xmin, xmax
#         if λmax < λmin
#             λmax += 360
#         end
#         x0, x1 = lon2x(λmin), lon2x(λmax)
#         y0, y1 = lat2y(ymin), lat2y(ymax)
#     elseif crs == :webmercator
#         x0, y0, x1, y1 = xmin, ymin, xmax, ymax
#     else
#         error("Unsupported crs: $crs (use :wgs84 or :webmercator)")
#     end

#     w = abs(x1 - x0)
#     h = abs(y1 - y0)
#     vw, vh = viewport
#     if w <= 0 || h <= 0 || vw <= 0 || vh <= 0
#         return minzoom
#     end

#     # meters per pixel required to fit bbox
#     req_m_per_px = max(w / vw, h / vh)

#     # meters per pixel at zoom z: WORLD_M / tilepixels(z)
#     # Find largest z such that mpp(z) <= req_m_per_px  ->  2^z >= WORLD_M / (tile * req_m_per_px)
#     z = floor(Int, log2(WORLD_M / (tile * req_m_per_px)))

#     clamp(z, minzoom, maxzoom)
# end
