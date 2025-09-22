module Wildfires

using GLMakie, GeoMakie, Tyler, TileProviders, MapTiles, GeoJSON, DataFrames
using Tyler: ElevationProvider
using Makie: AbstractAxis

import Proj
import GMT
import GeometryOps as GO
import GeoInterface as GI

include("Data.jl")

#-----------------------------------------------------------------------------# coordinate transformations
lonlat2webmercator(x::Tuple{Real, Real}) = MapTiles.project(x, MapTiles.wgs84, MapTiles.web_mercator)
lonlat2webmercator(x) = map(lonlat2webmercator, collect(x))

webmercator2lonlat(x::Tuple{Real, Real}) = MapTiles.project(x, MapTiles.web_mercator, MapTiles.wgs84)
webmercator2lonlat(x) = map(webmercator2lonlat, collect(x))

const R = 6378137.0  # WGS84 radius in meters

# Conversion functions
mercator_to_lon(x) = x / R * 180 / π
mercator_to_lat(y) = atan(sinh(y / R)) * 180 / π

lon_to_mercator(lon) = lon * π / 180 * R
lat_to_mercator(lat) = log(tan((90 + lat) * π / 360)) * R

function mercator_axislabels!(ax::Axis)
    ax.xtickformat = xs -> ["$(round(mercator_to_lon(x); digits=2))°" for x in xs]
    ax.ytickformat = ys -> ["$(round(mercator_to_lat(y); digits=2))°" for y in ys]
    return ax
end

#-----------------------------------------------------------------------------# Makie plot interactions
# Get observables for mouse coordinates in web mercator and WGS84
function get_mouse_coords(ax::AbstractAxis)
    web_mercator = Observable((x=0f0, y=0f0))
    lon_lat = Observable((x=0f0, y=0f0))
    register_interaction!(ax, :get_mouse_web_mercator) do event::MouseEvent, axis
        data = MapTiles.project(event.data, MapTiles.web_mercator, MapTiles.wgs84)
        web_mercator[] = (x = event.data[1], y = event.data[2])
        lon_lat[] = (x = data[1], y = data[2])
    end
    (; web_mercator, lon_lat)
end

#-----------------------------------------------------------------------------# add_marshall_perimeter!
function add_marshall_perimeter!(ax::AbstractAxis, color=:red)
    data = Data.marshall()
    data2 = GO.reproject(data, "+proj=webmerc +datum=WGS84")

    # coords = Data.marshall().coordinates
    # coords2 = lonlat2webmercator(coords)
    # geom = GeoJSON.MultiPolygon(coordinates = coords2)
    poly!(ax, data2, color=(color, .2), strokecolor=color, strokewidth=1)
end

#-----------------------------------------------------------------------------# Map
@kwdef struct Map
    location::Observable{Rect2f} = Observable(get_location("Superior, CO"))
end

#-----------------------------------------------------------------------------# get_location
# Get Rect2f from a location string (or GMTdataset)
# TODO: set path for cache
get_location(x::AbstractString) = get_location(GMT.geocoder(x))

function get_location(gmt::GMT.GMTdataset; delta = 0.2)
    x, y = Proj.Transformation(gmt.proj4, "WGS84")(gmt.data...)
    Rect2f(y - delta/2, x - delta/2, delta, delta)
end



#-----------------------------------------------------------------------------# init_map
function init_map(loc = "Superior, CO";
        provider::Provider = TileProviders.OpenStreetMap(), #TileProviders.Google(:terrain),
        size = (1200, 1000)
    )
    fig = Figure(; size)
    ax = Axis(fig[1, 1], aspect = DataAspect(),
        xtickformat = xs -> ["$(round(mercator_to_lon(x); digits=2))°" for x in xs],
        ytickformat = ys -> ["$(round(mercator_to_lat(y); digits=2))°" for y in ys]
    )
    # ax = GeoAxis(fig[1, 1], dest="+proj=webmerc +datum=WGS84", aspect = DataAspect())
    m = Tyler.Map(get_location(loc); provider, figure = fig, axis = ax)
    wait(m)
    display(fig)

    add_marshall_perimeter!(ax)

    mouse_coords = get_mouse_coords(ax)
    return (; loc, m, provider, fig, ax, mouse_coords)
end

end
