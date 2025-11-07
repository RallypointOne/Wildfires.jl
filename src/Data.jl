module Data

import GeoInterface as GI
import GeometryOps as GO
import NaturalEarth
import MapTiles
import TileProviders
import ..Wildfires
import ImageMagick
import CSV
import Rasters

using GeoJSON, JSON3, Dates, DataFrames, HTTP, Extents, ColorTypes, FixedPointNumbers

#-----------------------------------------------------------------------------# Constants
const MARSHALL_FIRE_EXTENT = Extent(X=(-105.3, -105.1), Y=(39.9, 40.1))

#-----------------------------------------------------------------------------# AbstractData
abstract type AbstractData end
Base.getproperty(o::T, x::Symbol) where {T <: AbstractData} = x === :data ? get(o) : getfield(o, x)


abstract type StaticData <: AbstractData end
abstract type DynamicData <: AbstractData end
abstract type ForecastData <: DynamicData end


#-----------------------------------------------------------------------------# NASA FIRMS
# https://firms.modaps.eosdis.nasa.gov/active_fire/
@kwdef struct FIRMSActiveFiresCSV <: DynamicData
    region = "USA_contiguous_and_Hawaii"
    period::String = "24h"  # "24h", "48h", "7d"
    satellite::String = "MODIS_C6_1"
end
function Base.get(data::FIRMSActiveFiresCSV)
    url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/$(lowercase(data.satellite))/csv/$(data.satellite)_$(data.region)_$(data.period).csv"
    res = HTTP.get(url)
    CSV.read(IOBuffer(res.body), DataFrame)
end


# https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Canada_7d.csv

# https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_USA_contiguous_and_Hawaii_24h.csv
# https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_USA_contiguous_and_Hawaii_48h.csv
# https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/[MAP_KEY]/all

#-----------------------------------------------------------------------------# WFIGS
# https://data-nifc.opendata.arcgis.com/datasets/nifc%3A%3Awfigs-current-interagency-fire-perimeters/about
struct WFIGSCurrentPerimeters <: DynamicData end
function Base.get(::WFIGSCurrentPerimeters)
    url = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson"
    res = HTTP.get(url)
    obj = GeoJSON.read(res.body)
    DataFrame(obj)
end

# https://data-nifc.opendata.arcgis.com/datasets/nifc::current-wildland-fire-incident-locations/about
struct WFIGSCurrentIncidents <: DynamicData end
function Base.get(::WFIGSCurrentIncidents)
    url = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations_Current/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson"
    res = HTTP.get(url)
    obj = GeoJSON.read(res.body)
    DataFrame(obj)
end

# Refreshed every 5 minutes.  Data is on 15 minute delay
# https://data-nifc.opendata.arcgis.com/datasets/nifc::wfigs-interagency-fire-perimeters/about
# By default, get Marshall Fire perimeter
@kwdef struct WFIGSHistoricalPerimeters <: DynamicData
    from::DateTime = DateTime(2021, 12, 30)
    to::DateTime = DateTime(2022, 1, 10)
    extent::Union{Nothing, Extent} = MARSHALL_FIRE_EXTENT
end
function Base.get(data::WFIGSHistoricalPerimeters)
    url = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query?" *
        "where=(attr_FireDiscoveryDateTime >= '$(Dates.format(data.from, "yyyy-mm-ddTHH:MM:SSZ"))' AND attr_FireDiscoveryDateTime <= '$(Dates.format(data.to, "yyyy-mm-ddTHH:MM:SSZ"))')" *
        (isnothing(data.extent) ? "" : "&geometry=$(data.extent.X[1]),$(data.extent.Y[1]),$(data.extent.X[2]),$(data.extent.Y[2])") *
        (isnothing(data.extent) ? "" : "&geometryType=esriGeometryEnvelope") *
        (isnothing(data.extent) ? "" : "&spatialRel=esriSpatialRelIntersects") *
        (isnothing(data.extent) ? "" : "&inSR=4326") *
        "&outFields=*" *
        "&f=geojson"
    res = HTTP.get(url)
    DataFrame(GeoJSON.read(res.body))
end

#-----------------------------------------------------------------------------# Marshall Fire Final Perimeter
struct MarshallFireFinalPerimeter <: StaticData end
function Base.get(::MarshallFireFinalPerimeter)
    file = joinpath(@__DIR__, "..", "data", "marshall.geojson")
    GeoJSON.read(file)
end

#-----------------------------------------------------------------------------# NaturalEarth
include(joinpath(@__DIR__, "gen", "natural_earth_datasets.jl"))

struct NaturalEarthData <: AbstractData
    file::Symbol
    function NaturalEarthData(file::Symbol)
        file in natural_earth_datasets || error("Invalid Natural Earth dataset symbol: $file")
        new(file)
    end
end
Base.get(data::NaturalEarthData) = NaturalEarth.naturalearth(string(data.file))

Coastlines(scale = 10) = NaturalEarthData(Symbol("ne_$(scale)m_coastline"))
Countries(scale = 10) = NaturalEarthData(Symbol("ne_$(scale)m_admin_0_countries"))
Land(scale = 10) = NaturalEarthData(Symbol("ne_$(scale)m_land"))
Lakes(scale = 10) = NaturalEarthData(Symbol("ne_$(scale)m_lakes"))
Rivers(scale = 10) = NaturalEarthData(Symbol("ne_$(scale)m_rivers_lake_centerlines"))
# # Bathymetry levels A through L correspond to 10000m

# _bathymetry_levels = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 200, 0]
# Bathymetry(x::Char = 'A') = NaturalEarthData(Symbol("ne_10m_bathymetry_$(x)_$(_bathymetry_levels[findfirst(==(x), 'A':'L')])"))

#-----------------------------------------------------------------------------# DigitalElevationModel
struct DigitalElevationModel <: StaticData
    extent::Extent
end
function Base.get(data::DigitalElevationModel)
    bbox = (data.extent.X[1], data.extent.Y[1], data.extent.X[2], data.extent.Y[2])
    res = HTTP.get("""https://tnmaccess.nationalmap.gov/api/v1/products?datasets=Digital%20Elevation%20Model%20(DEM)%201%20meter&bbox=$bbox&prodFormats=GeoTIFF""")
    file = tempname() * ".tif"
    write(file, res.body)
    Rasters.Raster(file)
end

#-----------------------------------------------------------------------------# TileProviderData
@kwdef struct TileProviderData <: StaticData
    provider = TileProviders.USGS(:USTopo)
    zoom::Int = clamp(TileProviders.min_zoom(provider) + 2, TileProviders.min_zoom(provider), TileProviders.max_zoom(provider))
    ext::Extent = Extent(X=(-180.0, 180.0), Y=(-85.051129, 85.0511))  # World
end
function Base.get(data::TileProviderData)
    grid = MapTiles.TileGrid(data.ext, data.zoom, MapTiles.wgs84)
    nrow = length(grid.grid.indices[1])
    ncol = length(grid.grid.indices[2])
    out = Matrix{RGB{N0f8}}(undef, 256 * nrow, 256 * ncol)

    # tiles = map(grid) do t
    #     url = TileProviders.geturl(data.provider, t.x, t.y, t.z)
    #     @info "Downloading tile: $url"
    #     res = HTTP.get(url)
    #     ImageMagick.readblob(res.body)
    # end

    # out =

    # collect(grid)
end


# @kwdef struct TileData <: StaticData
#     tile::MapTiles.Tile = MapTiles.Tile(Geo.geocode("Boulder, CO"), 10, MapTiles.wgs84)
#     provider = TileProviders.OpenStreetMap()
# end
# TileData(loc::AbstractString, zoom=10) = TileData(tile=MapTiles.Tile(Geo.geocode(loc), zoom, MapTiles.wgs84))

# function Base.get((; tile, provider)::TileData)
#     bbox = extent(tile, MapTiles.web_mercator)
#     url = TileProviders.geturl(provider, tile.x, tile.y, tile.z)
#     res = HTTP.get(url)
# #     tiles = MapTiles.tile_range((data.extent.X[1], data.extent.Y[1]), (data.extent.X[2], data.extent.Y[2]), data.zoom)
# end


#-----------------------------------------------------------------------------# TODOs
# HRRR


#-----------------------------------------------------------------------------# filtering to extent
# using NaturalEarth
# import GeometryOps as GO
# import GeoInterface as GI
# using Extents

# # Load all countries in the world
# world_fc = naturalearth("admin_0_countries", 10)
# world_geoms = world_fc.geometry
# # Construct a spatial tree of your geometries.
# # Note this only stores integer indices in the leaves not the geometries themselves.
# tree = GO.STRtree(world_geoms)
# # What extent do you care about?
# target_extent = Extents.Extent(X = (6.5886, 8.5886), Y = (46.5596, 48.5596))

# import GeometryOps.SpatialTreeInterface as STI
# # simple - get all potentially intersecting geoms
# # based purely on extent checks
# STI.query(tree, target_extent)

# # slightly more specialized - DIY and there is no allocation overhead
# # beyond what you do in the closure
# intersecting_indices = Int[]
# STI.depth_first_search(GO.intersects(target_extent), tree) do i
#     if GO.intersects(world_geoms[i], target_extent)
#         push!(intersecting_indices, i)
#     end
# end

# names = world_fc.NAME[intersecting_indices]

# API functions are
# query # high level simple interface for tree query
# depth_first_search # single tree
# dual_depth_first_search # two trees simultaneously for e.g. spatial join
# # see other interface functions in GeometryOps/src/utils/SpatialTreeInterface/interfaces.jl

end  # module
