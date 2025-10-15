module Wildfires

using GLMakie, GeoMakie, Tyler, TileProviders, MapTiles, GeoJSON, DataFrames, Extents, StyledStrings

using Makie: AbstractAxis
using H3: H3, API

import Proj
import GMT

import GeoInterface as GI
import GeometryOps as GO
import GeoFormatTypes as GFT


export
    Data,
    # h3 stuff:
    Cell, LatLon, Vertex

#-----------------------------------------------------------------------------# Ideas
# A Map has:
# 1. A base layer (tile provider)
# 2. Overlay layers (fire perimeters, lakes, rivers, roads, power lines, etc.)
# 3. A propagation model layer
#   a. Propagation model can be used for:
#     i. Forecasting spread from a real fire perimeter/start point.
#     ii. Simulating spread from a start point (click and watch animation).
#   b. Model data layer (wind, humidity, temperature, fuel, elevation, etdc.).  MODELS ARE STRICTLY TIED TO DATASETS
#  4. Smoke layer???



#-----------------------------------------------------------------------------# includes
include("geo.jl")
include("Data.jl")

#-----------------------------------------------------------------------------# get_extent
function get_extent(loc::AbstractString, delta=0.2)
    gmt = GMT.geocoder(loc)
    x, y = Proj.Transformation(gmt.proj4, "WGS84")(gmt.data...)
    Extents.Extent(X = (y - delta/2, y + delta/2), Y = (x - delta/2, x + delta/2))
end

polygon((;X, Y)::Extent) = GI.Polygon([[(X[1], Y[1]), (X[2], Y[1]), (X[2], Y[2]), (X[1], Y[2]), (X[1], Y[1])]])


#-----------------------------------------------------------------------------# coordinate transformations
const R = 6378137.0  # WGS84 radius in meters

# Conversion functions
mercator_to_lon(x) = x / R * 180 / π
mercator_to_lat(y) = atan(sinh(y / R)) * 180 / π

lon_to_mercator(lon) = lon * π / 180 * R
lat_to_mercator(lat) = log(tan((90 + lat) * π / 360)) * R

#-----------------------------------------------------------------------------# Mouse
struct Mouse
    axis::AbstractAxis
    coords::Observable{Point2d}
    left_click::Observable{Point2d}
    right_click::Observable{Point2d}

    function Mouse(axis::AbstractAxis)
        coords = Observable(Point2d(0, 0))
        left_click = Observable(Point2d(0, 0))
        right_click = Observable(Point2d(0, 0))
        :mouse_observables in keys(Makie.interactions(axis)) && Makie.deregister_interaction!(axis, :mouse_observables)
        register_interaction!(axis, :mouse_observables) do event::MouseEvent, axis
            coords[] = event.data
            if event.type === MouseEventTypes.leftclick
                left_click[] = event.data
                notify(left_click)
            elseif event.type === MouseEventTypes.rightclick
                right_click[] = event.data
                notify(right_click)
            end
        end
        new(axis, coords, left_click, right_click)
    end
end

coords_string(x::Point2d) = "lon = $(round(mercator_to_lon(x[1]); digits=3))°, lat = $(round(mercator_to_lat(x[2]); digits=3))°"

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
# @kwdef struct Map
#     location::Observable{Rect2f} = Observable(get_location("Superior, CO"))
#     mouse_coords::Observable{Point2f} = Observable(Point2f(0, 0))
# end

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
        provider = TileProviders.OpenStreetMap(), #TileProviders.Google(:terrain),
        size = (1200, 1000)
    )
    figure = Figure(; size)
    axis = Axis(figure[2, 2], aspect = DataAspect(),
        xtickformat = xs -> ["$(round(mercator_to_lon(x); digits=2))°" for x in xs],
        ytickformat = ys -> ["$(round(mercator_to_lat(y); digits=2))°" for y in ys],

    )
    mouse = Mouse(axis)

    # Title in the top left
    title = Label(figure[1, 1], "Wildfires.jl", fontsize=18)

    # Top row above map
    top = figure[1, 2] = GridLayout()
    Label(top[1,2], " Placeholder 1")
    Label(top[1,3], " Placeholder 2")

    # Create the map
    m = Tyler.Map(get_location(loc); provider, figure, axis)
    wait(m)  # wait for tiles to load

    # Add lon/lat position in bottom right corner
    textlabel!(axis, Point2f(1,0), text=@lift(coords_string($(mouse.coords))), shape=Rect2f(0, 0, 1, 1), text_align=(:right, :bottom), text_color=:white, background_color=(:black, 0.5), fontsize=12, space=:relative, cornerradius=0, offset=(-4, 4), strokewidth=0, font="Consolas")

    # left click
    scatter!(axis, mouse.left_click, color=:red, marker=:cross, markersize=20)

    # left side of map
    left = figure[2, 1] = GridLayout()

    # base_layer_ui = left[1,1] = GridLayout()
    # Label(base_layer_ui[1,1], "Base Layer")
    Menu(left[1,1], options=["Option 1", "Option 2", "Option 3"], default="Option 1")

    # Label(left[1,1], "Placeholder 1")
    Label(left[2,1], "Placeholder 2")
    Label(left[3,1], "Placeholder 3")

    # Layout
    # rowsize!(figure.layout, 1, Fixed(30))
    rowsize!(figure.layout, 2, Relative(4/5))
    colsize!(figure.layout, 2, Relative(4/5))
    # colsize!(figure.layout, 1, Relative(1/5))

    # Add Marshall fire perimeter
    add_marshall_perimeter!(axis)

    display(figure)

    return (; loc, m, provider, figure, title, top, left, axis, mouse)
end

end
