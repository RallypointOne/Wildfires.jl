# Resources:
# - https://data-nifc.opendata.arcgis.com



module Data
using GeoJSON, Dates, DataFrames, HTTP
import ..Wildfires

#-----------------------------------------------------------------------------# AbstractData
abstract type AbstractData end
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

#-----------------------------------------------------------------------------# Marshall Fire Final Perimeter
struct MarshallFireFinalPerimeter <: StaticData end
function Base.get(::MarshallFireFinalPerimeter)
    file = joinpath(@__DIR__, "..", "data", "marshall.geojson")
    GeoJSON.read(file)
end

end  # module
