module WildfiresMakieExt

using Wildfires
using Wildfires.LevelSet: LevelSetGrid, xcoords, ycoords
using Wildfires.SpreadModel: Trace

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

#-----------------------------------------------------------------------------# Burnout colormap
const _FIRE_CMAP = Makie.cgrad([:black, :red, :yellow, :white, :green], [0.0, 0.25, 0.5, 0.75, 1.0])

# Build a [-1, 1] value matrix: burned → negative, unburned → positive
function _fire_values(φ, t_ignite, t_now, t_r)
    vals = similar(φ)
    φ_max = maximum(x -> x > 0 ? x : zero(x), φ)
    φ_max = φ_max > 0 ? φ_max : one(φ_max)
    for j in axes(φ, 2), i in axes(φ, 1)
        t_ig = t_ignite[i, j]
        if isnan(t_ig)
            vals[i, j] = zero(eltype(vals))
        elseif isfinite(t_ig) && t_ig <= t_now
            vals[i, j] = -clamp((t_now - t_ig) / t_r, 0, 1)
        else
            vals[i, j] = clamp(φ[i, j] / φ_max, 0, 1)
        end
    end
    vals
end

#-----------------------------------------------------------------------------# fireplot / fireplot!
function Wildfires.fireplot!(ax, grid::LevelSetGrid;
        residence_time=nothing, frontcolor=:black, frontlinewidth=2.0,
        colormap=nothing)
    xs = collect(xcoords(grid))
    ys = collect(ycoords(grid))
    φ = collect(grid.φ)

    if residence_time !== nothing
        vals = _fire_values(φ, collect(grid.t_ignite), grid.t, residence_time)
        heatmap!(ax, xs, ys, vals; colormap=_FIRE_CMAP, colorrange=(-1, 1))
    else
        cmap = colormap === nothing ? :RdYlGn : colormap
        v = max(abs(minimum(φ)), abs(maximum(φ)))
        heatmap!(ax, xs, ys, φ; colormap=cmap, colorrange=(-v, v))
    end
    contour!(ax, xs, ys, φ; levels=[0.0], color=frontcolor, linewidth=frontlinewidth)
    ax
end

function Wildfires.fireplot(grid::LevelSetGrid;
        residence_time=nothing, frontcolor=:black, frontlinewidth=2.0,
        colormap=nothing)
    fig = Figure()
    ax = Axis(fig[1, 1])
    Wildfires.fireplot!(ax, grid; residence_time, frontcolor, frontlinewidth, colormap)
    fig
end

#-----------------------------------------------------------------------------# firegif
function Wildfires.firegif(path, trace::Trace, grid::LevelSetGrid;
        residence_time=nothing, framerate=15, frontcolor=:black, frontlinewidth=2.0,
        colormap=nothing)
    xs = collect(xcoords(grid))
    ys = collect(ycoords(grid))

    fig = Figure(size=(500, 450))
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x (m)", ylabel="y (m)")

    if residence_time !== nothing
        record(fig, path, eachindex(trace.stack); framerate) do idx
            empty!(ax)
            t, φ = trace.stack[idx]
            ax.title = "t = $(round(t, digits=1)) min"
            vals = _fire_values(φ, collect(grid.t_ignite), t, residence_time)
            heatmap!(ax, xs, ys, vals; colormap=_FIRE_CMAP, colorrange=(-1, 1))
            contour!(ax, xs, ys, φ; levels=[0.0], color=frontcolor, linewidth=frontlinewidth)
        end
    else
        cmap = colormap === nothing ? :RdYlGn : colormap
        φ_final = trace.stack[end][2]
        v = max(abs(minimum(φ_final)), abs(maximum(φ_final)))
        record(fig, path, eachindex(trace.stack); framerate) do idx
            empty!(ax)
            t, φ = trace.stack[idx]
            ax.title = "t = $(round(t, digits=1)) min"
            heatmap!(ax, xs, ys, φ; colormap=cmap, colorrange=(-v, v))
            contour!(ax, xs, ys, φ; levels=[0.0], color=frontcolor, linewidth=frontlinewidth)
        end
    end
    path
end

end # module
