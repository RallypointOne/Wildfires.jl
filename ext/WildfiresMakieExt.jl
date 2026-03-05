module WildfiresMakieExt

using Wildfires
using Wildfires.LevelSet: LevelSetGrid, xcoords, ycoords
using Wildfires.SpreadModels: Trace, RothermelModel, spread_rate_field!

using Contour
using Makie

# Grid φ is (ny, nx); Makie wants z[i,j] at (x[i], y[j]) → (nx, ny).
# Transpose at every Makie boundary.
_makie(M::AbstractMatrix) = collect(M')

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
    Makie.convert_arguments(P, collect(xcoords(g)), collect(ycoords(g)), _makie(g.φ))
end

#-----------------------------------------------------------------------------# Burnout colormap
const _FIRE_CMAP = Makie.cgrad([:black, :red, :yellow, :gray, :green], [0.0, 0.25, 0.5, 0.6, 1.0])

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
    φ = _makie(grid.φ)

    if residence_time !== nothing
        vals = _makie(_fire_values(collect(grid.φ), collect(grid.t_ignite), grid.t, residence_time))
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
            vals = _makie(_fire_values(φ, collect(grid.t_ignite), t, residence_time))
            heatmap!(ax, xs, ys, vals; colormap=_FIRE_CMAP, colorrange=(-1, 1))
            contour!(ax, xs, ys, _makie(φ); levels=[0.0], color=frontcolor, linewidth=frontlinewidth)
        end
    else
        cmap = colormap === nothing ? :RdYlGn : colormap
        φ_final = trace.stack[end][2]
        v = max(abs(minimum(φ_final)), abs(maximum(φ_final)))
        record(fig, path, eachindex(trace.stack); framerate) do idx
            empty!(ax)
            t, φ = trace.stack[idx]
            ax.title = "t = $(round(t, digits=1)) min"
            heatmap!(ax, xs, ys, _makie(φ); colormap=cmap, colorrange=(-v, v))
            contour!(ax, xs, ys, _makie(φ); levels=[0.0], color=frontcolor, linewidth=frontlinewidth)
        end
    end
    path
end

#-----------------------------------------------------------------------------# fireplot with RothermelModel components
"""
    fireplot(grid::LevelSetGrid, model::RothermelModel; wind=true, moisture=true, terrain=true, spread_rate=false, subsample=nothing, kwargs...)

Multi-panel fire visualization showing the fire state alongside environmental components
from the `RothermelModel`.

Toggle panels with keyword arguments:
- `wind=true` — wind speed heatmap with push-direction arrows
- `moisture=true` — 1-hr dead fuel moisture heatmap (%)
- `terrain=true` — terrain slope heatmap (°) with upslope arrows
- `spread_rate=false` — head-fire spread rate field (m/min)

The fire front contour (`φ = 0`) is overlaid on every panel for spatial reference.

Additional keyword arguments are forwarded to the fire state panel
(see [`fireplot`](@ref) for `residence_time`, `frontcolor`, `frontlinewidth`, `colormap`).

### Examples
```julia
using CairoMakie
fireplot(grid, model)
fireplot(grid, model; wind=false, spread_rate=true)
fireplot(grid, model; residence_time=0.005, terrain=false)
```
"""
function Wildfires.fireplot(grid::LevelSetGrid, model::RothermelModel;
        residence_time=nothing, frontcolor=:black, frontlinewidth=2.0, colormap=nothing,
        wind=true, moisture=true, terrain=true, spread_rate=false, subsample=nothing)
    panels = Symbol[:fire]
    wind && push!(panels, :wind)
    moisture && push!(panels, :moisture)
    terrain && push!(panels, :terrain)
    spread_rate && push!(panels, :spread_rate)

    npanels = length(panels)
    if npanels == 1
        return Wildfires.fireplot(grid; residence_time, frontcolor, frontlinewidth, colormap)
    end

    ncols = min(npanels, 2)
    nrows = ceil(Int, npanels / ncols)
    sub = something(subsample, max(1, min(size(grid)...) ÷ 15))

    fig = Figure(size=(550 * ncols, 500 * nrows))

    xs = collect(xcoords(grid))
    ys = collect(ycoords(grid))
    φ = _makie(grid.φ)

    for (idx, panel) in enumerate(panels)
        r = (idx - 1) ÷ ncols + 1
        c = (idx - 1) % ncols + 1
        gl = fig[r, c] = GridLayout()

        if panel == :fire
            ax = Axis(gl[1, 1], title="Fire State", aspect=DataAspect(),
                xlabel="x (m)", ylabel="y (m)")
            Wildfires.fireplot!(ax, grid; residence_time, frontcolor, frontlinewidth, colormap)
        elseif panel == :wind
            _plot_wind_panel!(gl, model.wind, grid, xs, ys, φ, sub)
        elseif panel == :moisture
            _plot_moisture_panel!(gl, model.moisture, grid, xs, ys, φ)
        elseif panel == :terrain
            _plot_terrain_panel!(gl, model.terrain, grid, xs, ys, φ, sub)
        elseif panel == :spread_rate
            _plot_spread_rate_panel!(gl, model, grid, xs, ys, φ)
        end
    end

    fig
end

#-----------------------------------------------------------------------------# Component panel helpers
function _plot_wind_panel!(gl, wind, grid, xs, ys, φ, sub)
    t = grid.t
    speed = Matrix{Float64}(undef, length(ys), length(xs))
    for j in eachindex(xs), i in eachindex(ys)
        s, _ = wind(t, xs[j], ys[i])
        speed[i, j] = s
    end

    ax = Axis(gl[1, 1], title="Wind Speed (km/h)", aspect=DataAspect(),
        xlabel="x (m)", ylabel="y (m)")
    hm = heatmap!(ax, xs, ys, _makie(speed); colormap=:Blues)
    Colorbar(gl[1, 2], hm)
    contour!(ax, xs, ys, φ; levels=[0.0], color=:red, linewidth=1.5)

    # Push-direction arrows (subsampled, unit length)
    si = 1:sub:length(ys)
    sj = 1:sub:length(xs)
    arrowlen = grid.dx * sub * 0.4
    ax_pts, ay_pts, au, av = Float64[], Float64[], Float64[], Float64[]
    for j in sj, i in si
        _, d = wind(t, xs[j], ys[i])
        push!(ax_pts, xs[j])
        push!(ay_pts, ys[i])
        push!(au, -cos(d) * arrowlen)
        push!(av, -sin(d) * arrowlen)
    end
    if !isempty(ax_pts)
        arrows2d!(ax, ax_pts, ay_pts, au, av; tipwidth=8, color=(:black, 0.6))
    end
    ax
end

function _plot_moisture_panel!(gl, moisture, grid, xs, ys, φ)
    t = grid.t
    d1 = Matrix{Float64}(undef, length(ys), length(xs))
    for j in eachindex(xs), i in eachindex(ys)
        fc = moisture(t, xs[j], ys[i])
        d1[i, j] = fc.d1 * 100
    end

    ax = Axis(gl[1, 1], title="1-hr Fuel Moisture (%)", aspect=DataAspect(),
        xlabel="x (m)", ylabel="y (m)")
    hm = heatmap!(ax, xs, ys, _makie(d1); colormap=:YlGnBu)
    Colorbar(gl[1, 2], hm)
    contour!(ax, xs, ys, φ; levels=[0.0], color=:red, linewidth=1.5)
    ax
end

function _plot_terrain_panel!(gl, terrain, grid, xs, ys, φ, sub)
    t = grid.t
    slope = Matrix{Float64}(undef, length(ys), length(xs))
    for j in eachindex(xs), i in eachindex(ys)
        s, _ = terrain(t, xs[j], ys[i])
        slope[i, j] = atand(s)
    end

    ax = Axis(gl[1, 1], title="Slope (°)", aspect=DataAspect(),
        xlabel="x (m)", ylabel="y (m)")
    hm = heatmap!(ax, xs, ys, _makie(slope); colormap=:YlOrBr)
    Colorbar(gl[1, 2], hm)
    contour!(ax, xs, ys, φ; levels=[0.0], color=:red, linewidth=1.5)

    # Upslope arrows (fire push direction, only where slope > 0)
    si = 1:sub:length(ys)
    sj = 1:sub:length(xs)
    arrowlen = grid.dx * sub * 0.4
    ax_pts, ay_pts, au, av = Float64[], Float64[], Float64[], Float64[]
    for j in sj, i in si
        s, a = terrain(t, xs[j], ys[i])
        if s > 0
            push!(ax_pts, xs[j])
            push!(ay_pts, ys[i])
            push!(au, -cos(a) * arrowlen)
            push!(av, -sin(a) * arrowlen)
        end
    end
    if !isempty(ax_pts)
        arrows2d!(ax, ax_pts, ay_pts, au, av; tipwidth=8, color=(:black, 0.6))
    end
    ax
end

function _plot_spread_rate_panel!(gl, model, grid, xs, ys, φ)
    F = similar(grid.φ, Float64)
    spread_rate_field!(F, model, grid)

    ax = Axis(gl[1, 1], title="Spread Rate (m/min)", aspect=DataAspect(),
        xlabel="x (m)", ylabel="y (m)")
    hm = heatmap!(ax, xs, ys, _makie(F); colormap=:inferno)
    Colorbar(gl[1, 2], hm)
    contour!(ax, xs, ys, φ; levels=[0.0], color=:white, linewidth=1.5)
    ax
end

#-----------------------------------------------------------------------------# fireplot3d
function Wildfires.fireplot3d!(ax, grid::LevelSetGrid, elevation::AbstractMatrix;
        residence_time=nothing, frontcolor=:black, frontlinewidth=2.0, colormap=nothing)
    xs = collect(xcoords(grid))
    ys = collect(ycoords(grid))
    elev = _makie(elevation)
    φ = _makie(grid.φ)

    if residence_time !== nothing
        color = _makie(_fire_values(collect(grid.φ), collect(grid.t_ignite), grid.t, residence_time))
        surface!(ax, xs, ys, elev; color=color, colormap=_FIRE_CMAP, colorrange=(-1, 1))
    else
        cmap = colormap === nothing ? :RdYlGn : colormap
        v = max(abs(minimum(φ)), abs(maximum(φ)))
        surface!(ax, xs, ys, elev; color=φ, colormap=cmap, colorrange=(-v, v))
    end

    # 3D fire front: extract φ=0 contour and lift to terrain surface
    _fire_front_3d!(ax, xs, ys, φ, elev, frontcolor, frontlinewidth)
    ax
end

function Wildfires.fireplot3d(grid::LevelSetGrid, elevation::AbstractMatrix;
        residence_time=nothing, frontcolor=:black, frontlinewidth=2.0, colormap=nothing)
    fig = Figure(size=(800, 600))
    ax = Axis3(fig[1, 1], xlabel="x (m)", ylabel="y (m)", zlabel="Elevation (m)")
    Wildfires.fireplot3d!(ax, grid, elevation; residence_time, frontcolor, frontlinewidth, colormap)
    fig
end

#-----------------------------------------------------------------------------# 3D fire front helpers
function _fire_front_3d!(ax, xs, ys, φ, elev, frontcolor, frontlinewidth)
    cl = Contour.contours(xs, ys, φ, [0.0])
    for level in Contour.levels(cl)
        for line in Contour.lines(level)
            cx, cy = Contour.coordinates(line)
            isempty(cx) && continue
            cz = [_interp_elev(elev, xs, ys, x, y) for (x, y) in zip(cx, cy)]
            lines!(ax, cx, cy, cz; color=frontcolor, linewidth=frontlinewidth)
        end
    end
end

function _interp_elev(elev, xs, ys, px, py)
    ix = clamp(searchsortedlast(xs, px), 1, length(xs) - 1)
    iy = clamp(searchsortedlast(ys, py), 1, length(ys) - 1)
    tx = clamp((px - xs[ix]) / (xs[ix+1] - xs[ix]), 0.0, 1.0)
    ty = clamp((py - ys[iy]) / (ys[iy+1] - ys[iy]), 0.0, 1.0)
    (1 - tx) * (1 - ty) * elev[ix, iy] +
        tx * (1 - ty) * elev[ix+1, iy] +
        (1 - tx) * ty * elev[ix, iy+1] +
        tx * ty * elev[ix+1, iy+1]
end

end # module
