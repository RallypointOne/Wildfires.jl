module WFIGS

using Extents, GeoJSON, Dates, DataFrames, HTTP
using ..Wildfires

@kwdef struct Query
    start::DateTime = now() - Day(7)
    stop::DateTime = now()
    extent::Extents.Extent = Wildfires.get_extent("Boulder, CO", 2.0)
end

function url(q::Query)
    out = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query?" *
        "where=poly_DateCurrent >= timestamp '#$(Dates.format(q.start_datetime, "yyyy-mm-dd HH:MM:ss"))' AND poly_DateCurrent <= timestamp '#$(Dates.format(q.end_datetime, "yyyy-mm-dd HH:MM:ss") )'"*
        "&outFields=*&outSR=4326&f=geojson"
end











end  # WFIGS module
