# Figures for the Marshall Fire hindcast. Included by marshall_fire.jl after
# the run, so it uses that script's globals: grid, model, fuel, structures,
# ∂x_h, ∂y_h, u, v, observed, history, T₀, x_ign, y_ign, crs, dem, DATA.

using CairoMakie

const FIGURES = joinpath(@__DIR__, "figures")
mkpath(FIGURES)

#-----------------------------------------------------------------------------# Shared layers
x_km = xnodes(grid, Center()) ./ 1e3
y_km = ynodes(grid, Center()) ./ 1e3
field2d(f) = interior(f)[:, :, 1]

# Elevation on the fire grid and a hillshade from the LANDFIRE gradient,
# sun from the northwest at 45°.
elevation = fire_field()
set!(elevation, raster_topography(dem, crs))
hx, hy = field2d(∂x_h), field2d(∂y_h)
light = (-0.5, 0.5, 0.707)
hillshade = @. (-hx * light[1] - hy * light[2] + light[3]) / sqrt(1 + hx^2 + hy^2)

arrival = field2d(model.t_ignition) ./ 3600            # hours after 18:00 UTC, Inf unburned
arrival_plot = replace(arrival, Inf => NaN)
observed_mask = field2d(observed)
footprints = field2d(structures.footprint_fraction)
footprint_plot = replace(footprints, 0.0 => NaN)
t_end = model.clock.time
hours(t) = Dates.format(T₀ + Second(round(Int, t)), "HH:MM")

function base_map!(ax)
    heatmap!(ax, x_km, y_km, hillshade; colormap = :grays, colorrange = (0.3, 1.1))
    heatmap!(ax, x_km, y_km, footprint_plot; colormap = [:gray30], nan_color = :transparent)
    contour!(ax, x_km, y_km, observed_mask; levels = [0.5], color = :black, linewidth = 1.5,
             linestyle = :dash)
    scatter!(ax, [x_ign / 1e3], [y_ign / 1e3]; marker = :star5, markersize = 18, color = :yellow,
             strokecolor = :black, strokewidth = 1)
    ax.xlabel = "km east of 105.182°W"
    ax.ylabel = "km north of 39.9575°N"
    ax.aspect = DataAspect()
    return ax
end

#-----------------------------------------------------------------------------# Inputs
fig = Figure(size = (1400, 900))

ax = Axis(fig[1, 1]; title = "Terrain (m)")
base_map!(ax)
contour!(ax, x_km, y_km, field2d(elevation); levels = 1600:20:1860, color = :sienna, linewidth = 0.8,
         labels = true, labelsize = 10)

ax = Axis(fig[1, 2]; title = "LANDFIRE fuel model (FBFM13; NB = not burnable)")
base_map!(ax)
codes = [string.(fuel.codes[1:end-1]); "NB"]
hm = heatmap!(ax, x_km, y_km, field2d(fuel.index); colormap = :tab20, colorrange = (1, length(codes)),
              alpha = 0.7)
Colorbar(fig[1, 3], hm; ticks = (1:length(codes), codes))

ax = Axis(fig[2, 1]; title = "Structures: mean plan dimension within 60 m (m)")
base_map!(ax)
plan = replace(field2d(structures.plan_dimension), 0.0 => NaN)
hm = heatmap!(ax, x_km, y_km, plan; colormap = :viridis, nan_color = :transparent, alpha = 0.8)
Colorbar(fig[2, 3], hm)

ax = Axis(fig[2, 2]; title = "HRRR 10 m wind at 18:00 UTC (m/s)")
base_map!(ax)
u₀, v₀ = field2d(u[Time(0.0)]), field2d(v[Time(0.0)])
hm = heatmap!(ax, x_km, y_km, hypot.(u₀, v₀); colormap = :thermal, alpha = 0.6)
step = 40
xs = x_km[1:step:end]; ys = y_km[1:step:end]
quiver = isdefined(Makie, :arrows2d!) ? Makie.arrows2d! : Makie.arrows!
quiver(ax, xs, ys, u₀[1:step:end, 1:step:end], v₀[1:step:end, 1:step:end];
       lengthscale = 0.03, color = :black)
Colorbar(fig[2, 4], hm)

save(joinpath(FIGURES, "inputs.png"), fig)

#-----------------------------------------------------------------------------# Arrival time
fig = Figure(size = (1000, 750))
ax = Axis(fig[1, 1]; title = "Fire arrival time, hours after 18:00 UTC; dashed: observed final perimeter")
base_map!(ax)
levels = 0:0.5:ceil(t_end / 3600)
cf = heatmap!(ax, x_km, y_km, arrival_plot; colormap = cgrad(:inferno, length(levels) - 1; categorical = true),
              colorrange = extrema(levels), nan_color = :transparent)
ignited = isfinite.(structures.t_ignition)
scatter!(ax, structures.x[ignited] ./ 1e3, structures.y[ignited] ./ 1e3;
         color = structures.t_ignition[ignited] ./ 3600, colormap = :inferno, colorrange = extrema(levels),
         markersize = 5, strokecolor = :white, strokewidth = 0.5)
Colorbar(fig[1, 2], cf; label = "hours")
save(joinpath(FIGURES, "arrival_time.png"), fig)

#-----------------------------------------------------------------------------# Time series
fig = Figure(size = (900, 900))
th = history.t ./ 3600
ax = Axis(fig[1, 1]; ylabel = "burned area (km²)", title = "Marshall Fire hindcast, 30 December 2021")
lines!(ax, th, history.area ./ 1e6; linewidth = 2, label = "model")
hlines!(ax, [count(observed_mask .> 0.5) * Δx_fire * Δy_fire / 1e6]; color = :black, linestyle = :dash,
        label = "observed final perimeter")
axislegend(ax; position = :lt)
ax = Axis(fig[2, 1]; ylabel = "structures ignited")
lines!(ax, th, history.structures; linewidth = 2, label = "model")
hlines!(ax, [1084]; color = :black, linestyle = :dash, label = "observed destroyed")
axislegend(ax; position = :lt)
ax = Axis(fig[3, 1]; ylabel = "structure heat release (GW)")
lines!(ax, th, history.hrr ./ 1e9; linewidth = 2)
ax = Axis(fig[4, 1]; ylabel = "10 m wind at ignition (m/s)", xlabel = "hours after 18:00 UTC")
lines!(ax, th, history.wind; linewidth = 2)
save(joinpath(FIGURES, "timeseries.png"), fig)

#-----------------------------------------------------------------------------# Animation
# The arrival-time field holds the whole history: at time t the burned region
# is arrival ≤ t and the front is its boundary.
fig = Figure(size = (1000, 750))
title = Observable(hours(0.0) * " UTC")
ax = Axis(fig[1, 1]; title)
base_map!(ax)
frame_t = Observable(0.0)
burned_plot = lift(t -> replace(x -> x * 3600 <= t ? 1.0 : NaN, arrival), frame_t)
heatmap!(ax, x_km, y_km, burned_plot; colormap = [:orangered], nan_color = :transparent, alpha = 0.6)
contour!(ax, x_km, y_km, arrival_plot; levels = lift(t -> [t / 3600], frame_t), color = :red, linewidth = 2)
burning_color = lift(frame_t) do t
    [tᵢ <= t ? (:red, 1.0) : (:black, 0.0) for tᵢ in structures.t_ignition]
end
scatter!(ax, structures.x ./ 1e3, structures.y ./ 1e3; color = burning_color, markersize = 5)

record(fig, joinpath(FIGURES, "propagation.gif"), 0:600:t_end; framerate = 6) do t
    frame_t[] = t
    title[] = @sprintf("%s UTC   burned %.1f km²   %d structures ignited   wind at ignition %.0f m/s",
                       hours(t), count(x -> x * 3600 <= t, arrival) * Δx_fire * Δy_fire / 1e6,
                       count(<=(t), structures.t_ignition), wind_at_ignition(t))
end

@info "Figures written to $FIGURES" readdir(FIGURES)
