module WildfiresMakieExt

using Wildfires
using Wildfires.LevelSet: LevelSetGrid, xcoords, ycoords

using Makie

#-----------------------------------------------------------------------------# convert_arguments for standard Makie plot types
"""
    heatmap(grid::LevelSetGrid)
    contour(grid::LevelSetGrid)
    contourf(grid::LevelSetGrid)
    surface(grid::LevelSetGrid)

Standard Makie plot types work on `LevelSetGrid` with correct spatial coordinates.
The plotted values are the level set function `φ`.
"""
function Makie.convert_arguments(P::Type{<:Union{Makie.Heatmap,Makie.Contour,Makie.Contourf,Makie.Surface}}, g::LevelSetGrid)
    Makie.convert_arguments(P, collect(xcoords(g)), collect(ycoords(g)), collect(g.φ))
end

#-----------------------------------------------------------------------------# fireplot / fireplot!
function Wildfires.fireplot!(ax, grid::LevelSetGrid; colormap=:RdYlGn, frontcolor=:black, frontlinewidth=2.0)
    xs = collect(xcoords(grid))
    ys = collect(ycoords(grid))
    φ = collect(grid.φ)
    v = max(abs(minimum(φ)), abs(maximum(φ)))
    heatmap!(ax, xs, ys, φ; colormap, colorrange=(-v, v))
    contour!(ax, xs, ys, φ; levels=[0.0], color=frontcolor, linewidth=frontlinewidth)
    ax
end

function Wildfires.fireplot(grid::LevelSetGrid; colormap=:RdYlGn, frontcolor=:black, frontlinewidth=2.0)
    fig = Figure()
    ax = Axis(fig[1, 1])
    Wildfires.fireplot!(ax, grid; colormap, frontcolor, frontlinewidth)
    fig
end

end # module
