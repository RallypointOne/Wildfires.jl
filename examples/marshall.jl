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
using Serialization

import GeoInterface as GI
import GeometryOps as GO
import GeoFormatTypes as GFT
import GeoJSON


#-----------------------------------------------------------------------------# Marshall Fire
marshall = get(Data.MarshallFireFinalPerimeter())

# Add some padding around the extent of the fire perimeter
ext = Extents.grow(GI.extent(marshall), 0.1)


#-----------------------------------------------------------------------------# H3 Cells
res = 12

cells = sort!(h3cells(ext, res), by = x -> x.index)
marshall_cells = h3cells(marshall, res)


#-----------------------------------------------------------------------------# DataFrame of H3 Cells
df_path = abspath(joinpath(@__DIR__, "..", "data", "marshall_h3_$(res)_landfire.jld"))

if isfile(df_path)
    df = deserialize(df_path)
else
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

    landfire = Raster(landfire_file)
    landfire_bands = [landfire[Band=i].refdims[1][1] for i in 1:size(landfire, 3)]
    landfire_dict = Dict(band => landfire[Band=i] for (i, band) in enumerate(landfire_bands))

    @showprogress for (k, r) in landfire_dict
        T = eltype(r)
        df[!, k] = [T[] for _ in 1:nrow(df)]
        for ((lon, lat), val) in zip(Iterators.product(r.dims...), r)
            cell = H3Cell(LonLat(lon, lat), res)
            i = get(cell_idx, cell, nothing)
            if isnothing(i)
                continue
            else
                push!(df[i, k], val)
            end
        end
    end

    serialize(df_path, df)
    df = deserialize(df_path)
end







#-----------------------------------------------------------------------------# make_axis
# Default axis for maps
function make_axis(grid_pos; kw...)
    axis = GeoAxis(grid_pos, dest="+proj=webmerc +datum=WGS84", panbutton=Mouse.left)
    deregister_interaction!(axis, :rectanglezoom)  # Needed to enable panning with left mouse button
    hidedecorations!(axis, label=false, ticklabels=false, ticks=false, grid=true)
    axis
end

function base_map(provider = Tyler.ElevationProvider(nothing))
    figure = Figure()
    axis = make_axis(figure[1, 1])
    map = Tyler.Map(ext; provider, figure, axis)
    display(figure)
    return (; figure, axis, map)
end

#-----------------------------------------------------------------------------# Figure 1
fig, ax, m = base_map()
sleep(1)
poly!(ax, marshall, color=(:red, 0.1), strokecolor=:red, strokewidth=1)
# lines!(ax, GI.centroid.(marshall_cells), color=:blue)


# fig = Figure()
# ax = make_axis(fig[1, 1])
# poly!(ax, marshall, color=(:red, 0.2), strokecolor=:red, strokewidth=1)
# fig
# end
