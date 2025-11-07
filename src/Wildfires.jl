module Wildfires

using GLMakie
using GeoMakie
using Tyler
using TileProviders
using MapTiles
using GeoJSON
using DataFrames
using Extents
using StyledStrings
using OSMGeocoder

using Makie: AbstractAxis

import Proj
import GeoInterface as GI
import GeometryOps as GO
import GeoFormatTypes as GFT
import GeometryBasics as GB


export Data



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


# #-----------------------------------------------------------------------------# includes
# # include("geo.jl")
include("Data.jl")

# #-----------------------------------------------------------------------------# coordinate transformations
# const R = 6378137.0  # WGS84 radius in meters

# # Conversion functions
# mercator_to_lon(x) = x / R * 180 / π
# mercator_to_lat(y) = atan(sinh(y / R)) * 180 / π

# lon_to_mercator(lon) = lon * π / 180 * R
# lat_to_mercator(lat) = log(tan((90 + lat) * π / 360)) * R

# #-----------------------------------------------------------------------------# Mouse
# struct Mouse
#     axis::AbstractAxis
#     coords::Observable{Point2d}
#     left_click::Observable{Point2d}
#     right_click::Observable{Point2d}

#     function Mouse(axis::AbstractAxis)
#         coords = Observable(Point2d(0, 0))
#         left_click = Observable(Point2d(0, 0))
#         right_click = Observable(Point2d(0, 0))
#         :mouse_observables in keys(Makie.interactions(axis)) && Makie.deregister_interaction!(axis, :mouse_observables)
#         register_interaction!(axis, :mouse_observables) do event::MouseEvent, axis
#             coords[] = event.data
#             if event.type === MouseEventTypes.leftclick
#                 left_click[] = event.data
#                 notify(left_click)
#             elseif event.type === MouseEventTypes.rightclick
#                 right_click[] = event.data
#                 notify(right_click)
#             end
#         end
#         new(axis, coords, left_click, right_click)
#     end
# end

# coords_string(x::Point2d) = "lon = $(round(mercator_to_lon(x[1]); digits=3))°, lat = $(round(mercator_to_lat(x[2]); digits=3))°"

#-----------------------------------------------------------------------------# add_marshall_perimeter!
function add_marshall_perimeter!(ax::GeoAxis, color=:red)
    data = Data.MarshallFireFinalPerimeter().data
    poly!(ax, data, color=(color, .2), strokecolor=color, strokewidth=1)
end
function add_marshall_perimeter!(ax::Axis, color=:red)
    data = Data.MarshallFireFinalPerimeter().data
    data2 = GO.reproject(data, "+proj=webmerc +datum=WGS84")
    poly!(ax, data2, color=(color, .2), strokecolor=color, strokewidth=1)
end

#-----------------------------------------------------------------------------# init_map
function init_map(ext::Extents.Extent = GI.extent(Data.MarshallFireFinalPerimeter().data);
        title = "Wildfire.jl Map Viewer",
        provider = TileProviders.OpenStreetMap(), #TileProviders.Google(:terrain),
        figure = Figure(size=(1200, 1000))
    )
    navbar = Box(figure[1, 1:2], color=:lightgray, height=40)
    Label(figure[1, 1:2][1,1], title, fontsize=18, halign=:left)
    Label(figure[1, 1:2][1,2], "(ctrl + click to reset view)", fontsize=12, halign=:right)

    sidebar = Box(figure[2, 1], color=:lightgray, width=200)
    axis = GeoAxis(figure[2, 2], dest="+proj=webmerc +datum=WGS84", panbutton=Mouse.left)
    deregister_interaction!(axis, :rectanglezoom)  # Needed to enable panning with left mouse button
    hidedecorations!(axis, label=false, ticklabels=false, ticks=false, grid=true)

    m = Tyler.Map(ext; figure, axis)

    display(figure)
    (; figure, axis)
end

end  # module Wildfires
