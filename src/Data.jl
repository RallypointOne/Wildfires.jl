module Data

import GeoInterface as GI
import GeometryOps as GO
import NaturalEarth

using GeoJSON, Dates, DataFrames, HTTP, Extents

#-----------------------------------------------------------------------------# AbstractData
abstract type AbstractData end
function Base.getproperty(o::T, x::Symbol) where {T <: AbstractData}
    x === :data ? get(o) : getfield(o, x)
end

abstract type StaticData <: AbstractData end
abstract type DynamicData <: AbstractData end


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
    extent::Extent = Extent(X=(-105.3, -105.1), Y=(39.9, 40.1))
end
function Base.get(data::WFIGSHistoricalPerimeters)
    url = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query?" *
        "where=(attr_FireDiscoveryDateTime >= '$(Dates.format(data.from, "yyyy-mm-ddTHH:MM:SSZ"))' AND attr_FireDiscoveryDateTime <= '$(Dates.format(data.to, "yyyy-mm-ddTHH:MM:SSZ"))')" *
        "&geometry=$(data.extent.X[1]),$(data.extent.Y[1]),$(data.extent.X[2]),$(data.extent.Y[2])" *
        "&geometryType=esriGeometryEnvelope" *
        "&spatialRel=esriSpatialRelIntersects" *
        "&inSR=4326" *
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
Land(scale=10) = NaturalEarthData(Symbol("ne_$(scale)m_land"))
Lakes(scale=10) = NaturalEarthData(Symbol("ne_$(scale)m_lakes"))
Rivers(scale=10) = NaturalEarthData(Symbol("ne_$(scale)m_rivers_lake_centerlines"))
# # Bathymetry levels A through L correspond to 10000m

# _bathymetry_levels = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000, 200, 0]
# Bathymetry(x::Char = 'A') = NaturalEarthData(Symbol("ne_10m_bathymetry_$(x)_$(_bathymetry_levels[findfirst(==(x), 'A':'L')])"))

#-----------------------------------------------------------------------------# TODOs
# HRRR


end  # module
