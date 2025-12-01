using Wildfires
using GlobalGrids
using Extents
using GeoMakie, GLMakie
using Tyler
using TileProviders
using MapTiles
using Landfire
using DataFrames
using ProgressMeter
using Scratch
using Rasters
using SQLite
using DBInterface

import GeoInterface as GI
import GeometryOps as GO
import GeoFormatTypes as GFT
import GeoJSON

#-----------------------------------------------------------------------------# Database Connection
# db_path = joinpath(@__DIR__, "..", "data", "wildfire_db.sqlite")
# db = SQLite.DB(db_path)

#-----------------------------------------------------------------------------# Marshall Fire
# https://en.wikipedia.org/wiki/Marshall_Fire

marshall = get(Data.MarshallFireFinalPerimeter())

# Add some padding around the extent of the fire perimeter
ext = Extents.grow(GI.extent(marshall), 0.1)

# Two ignition points
ignition_points = [LonLat(-105.231, 39.955)]

#-----------------------------------------------------------------------------# H3 Cells
res = 11

cells = sort!(h3cells(ext, res), by = x -> x.index)
marshall_cells = h3cells(marshall, res)


#-----------------------------------------------------------------------------# DataFrame of H3 Cells
df_path = abspath(joinpath(@__DIR__, "..", "data", "marshall_h3_$(res)_landfire.jld"))

df = DataFrame()
df[!, :cells] = cells
df[!, :centroid] = GI.centroid.(cells)
df[!, :marshall] = in.(cells, Ref(marshall))

cell_idx = Dict(cell => i for (i, cell) in enumerate(cells))

#-----------------------------------------------------------------------------# Landfire Datasets
landfire_data = Scratch.get_scratch!(Wildfires, "landfire_data")
landfire_zip = joinpath(landfire_data, "landfire_products.zip")
landfire_products = filter!(Landfire.products(conus=true)) do p
    p.layer_name ∉ ("MF_F40FA24", "MF_FVCFA24", "MF_FVHFA24", "220ROADS_20")
end
landfire_products_df = DataFrame(landfire_products)

if !isfile(landfire_zip)
    Landfire.download(landfire_products, ext; output_projection="4326", dest=landfire_zip)
    run(`unzip -o $landfire_zip -d $landfire_data`)
end

landfire_file = filter(endswith(".tif"), readdir(landfire_data; join=true))[1]

landfire = Raster(landfire_file, checkmem = false)
landfire_bands = [landfire[Band=i].refdims[1][1] for i in 1:size(landfire, 3)]
landfire_dict = Dict(band => landfire[Band=i] for (i, band) in enumerate(landfire_bands))

#-----------------------------------------------------------------------------# make_axis
# Default axis for maps
function make_axis(grid_pos; kw...)
    axis = GeoAxis(grid_pos; dest="+proj=webmerc +datum=WGS84", panbutton=Mouse.left, kw...)
    deregister_interaction!(axis, :rectanglezoom)  # Needed to enable panning with left mouse button
    hidedecorations!(axis, label=false, ticklabels=false, ticks=false, grid=true)
    axis
end

t = Observable(0)
title = @lift string("t = ", $t)

function base_map()
    figure = Figure(;size=(1200, 1000))
    axis = make_axis(figure[1, 1], title = title)
    map = Tyler.Map(ext; figure, axis)
    display(figure)
    return (; figure, axis, map)
end

#-----------------------------------------------------------------------------# Cellular automata
fire_cells = Observable(H3Cell.(ignition_points, res))

fire_color = Observable([(:red, 1.0) for _ in ignition_points])

fig, ax, m = base_map()
sleep(3)  # Let tiles load
poly!(ax, marshall, color=(:red, 0.05), strokecolor=:red, strokewidth=1)
scatter!(ax, ignition_points, color = :black, markersize=8)
text!(ax, ignition_points, text = "Ignition", align = (:right, :center), color = :black, fontsize=12, offset = (-5, 0))
poly!(ax, fire_cells, color = :red)
# poly!(ax, fire_boundary_cells, color = (:red, 0.5), strokecolor=:red, strokewidth=1)
display(fig)

landfire_elevation = landfire_dict["US_ELEV2020"]

function spread!(wind = (direction° = 90.0, speed = 5.0))
    t[] += 1
    fire_color[] = map(fire_color[]) do (color, alpha)
        (color, max(alpha - 0.05, 0.1))
    end
    out = H3Cell[]

    _fire_cells = fire_cells[]

    for cell in _fire_cells
        neighbors = GlobalGrids.grid_ring(cell, 1)
        for candidate in neighbors
            candidate in _fire_cells && continue
            ll = GI.centroid(cell)
            candidate_ll = GI.centroid(candidate)
            azimuth = GlobalGrids.azimuth(ll, candidate_ll)
            wind_factor = wind.speed * cosd(azimuth - wind.direction°) * 2rand()

            elev = landfire_elevation[X(Near(ll.lon)), Y(Near(ll.lat))]
            elev_candidate = landfire_elevation[X(Near(candidate_ll.lon)), Y(Near(candidate_ll.lat))]
            elev_factor = elev_candidate / elev

            if (wind_factor > 0.4 || elev_factor > 1.01) && rand() < 0.7
                push!(out, candidate)
            end
        end
    end
    append!(fire_cells[], out)
    notify(fire_cells)
end

while true
    sleep(2)
    spread!()
end
