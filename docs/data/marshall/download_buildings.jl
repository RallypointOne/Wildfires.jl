#--------------------------------------------------------------------------------# OpenStreetMap building footprints for the Marshall Fire domain
#
# Queries the Overpass API for every way tagged `building` inside the LANDFIRE
# tile extent and writes them as a GeoJSON FeatureCollection of polygons in
# WGS84, with the OSM `building` tag value as the `building` property.
#
# Run from the docs environment:
#   julia --project=docs docs/data/marshall/download_buildings.jl
#
# Output: buildings.geojson
#--------------------------------------------------------------------------------

using Downloads, JSON3

const EXTENT = (south = 39.923, west = -105.244, north = 39.992, east = -105.121)
const OUTPATH = joinpath(@__DIR__, "buildings.geojson")

query = """
[out:json][timeout:300];
way["building"]($(EXTENT.south),$(EXTENT.west),$(EXTENT.north),$(EXTENT.east));
out geom;
"""

@info "Querying Overpass" EXTENT
body = IOBuffer()
Downloads.request("https://overpass-api.de/api/interpreter";
                  method = "POST", input = IOBuffer("data=" * query), output = body)
elements = JSON3.read(take!(body)).elements

features = Any[]
for way in elements
    ring = [[p.lon, p.lat] for p in way.geometry]
    length(ring) >= 4 && ring[1] == ring[end] || continue
    push!(features, (type = "Feature",
                     properties = (osm_id = way.id, building = get(way.tags, :building, "yes")),
                     geometry = (type = "Polygon", coordinates = [ring])))
end

open(OUTPATH, "w") do io
    JSON3.write(io, (type = "FeatureCollection", features = features))
end
@info "Wrote $(length(features)) building polygons to $OUTPATH"
