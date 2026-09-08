# HRRR analyses into a Breeze atmosphere over the Marshall Fire domain
#
# Step 1 of the fire–atmosphere coupling: the hourly HRRR pressure-level
# analyses (download_hrrr_levels.jl) give the initial state and a Davies
# relaxation toward the analysis in a lateral sponge, the way WPS and
# real.exe feed WRF. No fire yet: this run shows what the 200 m compressible
# core does with the analysis over the real terrain, and how far its
# near-surface wind departs from the 3 km product it is nudged toward.
#
# Run with:  julia --project=examples/marshall examples/marshall/atmosphere.jl

using Wildfires
using Oceananigans

using Oceananigans.Grids: xnodes, ynodes, znode, Center
using Oceananigans.OutputReaders: Time
using Breeze
using Breeze.TerrainFollowingDiscretization
using Rasters, ArchGDAL
using Dates, Printf, Statistics

include("domain.jl")
grid = atmosphere_grid

#-----------------------------------------------------------------------------# HRRR pressure levels
# Bands are the isobaric levels 1000:-25:300 hPa; HGT gives each level's
# geopotential height above sea level, the same datum as the grid's z. The
# lowest levels sit below the 1.6–1.8 km terrain and are HRRR's own
# below-ground extrapolation.
const LEVELS = 1000:-25:300
stamps = T₀:Hour(1):STOP
times = [Dates.value(Second(t - T₀)) for t in stamps]
hrrr(var, t) = Raster(joinpath(DATA, "hrrr_$(var)_$(Dates.format(t, "HHMM")).tif"))

# GDAL's GRIB driver reports temperature in °C.
κ = 0.2857                                        # R_d / c_p
exner⁻¹ = reshape([(1e5 / (100p))^κ for p in LEVELS], 1, 1, :)
potential_temperature(T) = Raster((parent(read(T)) .+ 273.15) .* exner⁻¹, dims(T); crs = Rasters.crs(T))

heights = [hrrr("HGT", t) for t in stamps]
u_series = raster_column_series([hrrr("UGRD", t) for t in stamps], heights, times, grid, crs)
v_series = raster_column_series([hrrr("VGRD", t) for t in stamps], heights, times, grid, crs)
θ_series = raster_column_series([potential_temperature(hrrr("TMP", t)) for t in stamps], heights, times, grid, crs)
q_series = raster_column_series([hrrr("SPFH", t) for t in stamps], heights, times, grid, crs)

# Sea-level pressure for the hydrostatic reference from the analysis surface
# pressure and terrain height (hypsometric, lowest-level temperature).
surface_pressure = raster_sampler(hrrr("PRES_surface", T₀), crs)
surface_height = raster_sampler(hrrr("HGT_surface", T₀), crs)
T_surface = raster_sampler(hrrr("TMP", T₀)[:, :, 1], crs)     # 1000 hPa, below ground, °C
p_msl = mean(surface_pressure(x, y) * exp(9.81 * surface_height(x, y) / (287 * (T_surface(x, y) + 273.15)))
             for x in xnodes(grid, Center()), y in ynodes(grid, Center()))

# Domain-mean θ profile of the analysis, as a function of absolute z, for the
# reference state; replaced by the actual mean at set! time.
θ̄ = vec(mean(interior(θ_series[1]); dims = (1, 2)))
z̄ = [mean(znode(i, j, k, grid, Center(), Center(), Center()) for i in 1:size(grid, 1), j in 1:size(grid, 2))
     for k in 1:size(grid, 3)]
θ_profile(z) = z <= z̄[1] ? θ̄[1] : z >= z̄[end] ? θ̄[end] :
               (k = searchsortedlast(z̄, z); θ̄[k] + (θ̄[k+1] - θ̄[k]) * (z - z̄[k]) / (z̄[k+1] - z̄[k]))

#-----------------------------------------------------------------------------# Model
# Compressible split-explicit core, the one Breeze documents on terrain, with a
# Rayleigh sponge in the top 2 km and Davies relaxation toward the analysis
# over the outer 8 cells (1.6 km). Moisture is initialized but not relaxed.
#
# BLOCKED: the side walls are impenetrable, so the through-flow cannot enter
# and the interior wind collapses within minutes despite the sponge (see
# figures/atmosphere_wind.png). Open boundaries (NormalFlowBoundaryCondition
# on ρu and ρv fed from the analysis) are the missing piece. Also no surface
# layer (free slip), so the lowest level feels no drag.
dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(acoustic_cfl = 0.5,
                                                                sponge = UpperSponge(damping_rate = 0.1, depth = 2000));
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = p_msl,
                                reference_potential_temperature = θ_profile)
mask = lateral_sponge(grid; width = 8)
rate = 1 / 600
forcing = (u = relaxation_forcing(:u, u_series, rate; mask),
           v = relaxation_forcing(:v, v_series, rate; mask),
           θ = relaxation_forcing(:θ, θ_series, rate; mask))
model = AtmosphereModel(grid; dynamics, forcing,
                        advection = WENO(order = 5),
                        closure = SmagorinskyLilly(),
                        coriolis = FPlane(latitude = 39.9575))
# Velocities live on cell faces: average the centred analysis onto them.
face_x(A) = cat(A[1:1, :, :], (A[1:end-1, :, :] .+ A[2:end, :, :]) ./ 2, A[end:end, :, :]; dims = 1)
face_y(A) = cat(A[:, 1:1, :], (A[:, 1:end-1, :] .+ A[:, 2:end, :]) ./ 2, A[:, end:end, :]; dims = 2)
set!(model; ρ = HydrostaticallyBalancedDensity(surface_pressure = p_msl),
     θ = interior(θ_series[1]), qᵛ = interior(q_series[1]),
     u = face_x(interior(u_series[1])), v = face_y(interior(v_series[1])),
     compute_reference_state = true)
println(model)

#-----------------------------------------------------------------------------# Run
stop_time = 3600.0
simulation = Simulation(model; Δt = 2, stop_time)
conjure_time_step_wizard!(simulation; cfl = 0.5, max_Δt = 5)

# Lowest-level wind every ten minutes, at cell centres, and the analysis at
# the same level.
level1(f) = interior(f)[:, :, 1]
centre_x(A) = (A[1:end-1, :] .+ A[2:end, :]) ./ 2
centre_y(A) = (A[:, 1:end-1] .+ A[:, 2:end]) ./ 2
snapshots = (t = Float64[], u = Matrix{Float64}[], v = Matrix{Float64}[],
             u_hrrr = Matrix{Float64}[], v_hrrr = Matrix{Float64}[])
function snapshot!(sim)
    t = sim.model.clock.time
    push!(snapshots.t, t)
    push!(snapshots.u, centre_x(level1(sim.model.velocities.u)))
    push!(snapshots.v, centre_y(level1(sim.model.velocities.v)))
    push!(snapshots.u_hrrr, level1(u_series[Time(t)]))
    push!(snapshots.v_hrrr, level1(v_series[Time(t)]))
    w = sim.model.velocities.w
    @printf("%s UTC  Δt %.1f s  level-1 speed %.1f m/s (analysis %.1f)  max|w| %.2f m/s\n",
            Dates.format(T₀ + Second(round(Int, t)), "HH:MM"), sim.Δt,
            mean(hypot.(snapshots.u[end], snapshots.v[end])),
            mean(hypot.(snapshots.u_hrrr[end], snapshots.v_hrrr[end])), maximum(abs, w))
    return nothing
end
add_callback!(simulation, snapshot!, TimeInterval(600))
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
run!(simulation)

#-----------------------------------------------------------------------------# Figures
using CairoMakie
const FIGURES = joinpath(@__DIR__, "figures")
mkpath(FIGURES)
x_km = xnodes(grid, Center()) ./ 1e3
y_km = ynodes(grid, Center()) ./ 1e3
speed = hypot.(snapshots.u[end], snapshots.v[end])
speed_hrrr = hypot.(snapshots.u_hrrr[end], snapshots.v_hrrr[end])
crange = (0, max(maximum(speed), maximum(speed_hrrr)))

fig = Figure(size = (1400, 500))
ax = Axis(fig[1, 1]; title = "Breeze level-1 wind (100 m AGL), " * Dates.format(T₀ + Second(round(Int, stop_time)), "HH:MM") * " UTC",
          xlabel = "km east", ylabel = "km north", aspect = DataAspect())
heatmap!(ax, x_km, y_km, speed; colormap = :thermal, colorrange = crange)
ax = Axis(fig[1, 2]; title = "HRRR analysis at the same level", xlabel = "km east", aspect = DataAspect())
hm = heatmap!(ax, x_km, y_km, speed_hrrr; colormap = :thermal, colorrange = crange)
Colorbar(fig[1, 3], hm; label = "m/s")
ax = Axis(fig[1, 4]; title = "difference", xlabel = "km east", aspect = DataAspect())
d = speed .- speed_hrrr
hm = heatmap!(ax, x_km, y_km, d; colormap = :balance, colorrange = (-maximum(abs, d), maximum(abs, d)))
Colorbar(fig[1, 5], hm; label = "m/s")
save(joinpath(FIGURES, "atmosphere_wind.png"), fig)

fig = Figure(size = (800, 400))
ax = Axis(fig[1, 1]; xlabel = "minutes after 18:00 UTC", ylabel = "domain-mean level-1 speed (m/s)")
lines!(ax, snapshots.t ./ 60, [mean(hypot.(u, v)) for (u, v) in zip(snapshots.u, snapshots.v)]; label = "Breeze")
lines!(ax, snapshots.t ./ 60, [mean(hypot.(u, v)) for (u, v) in zip(snapshots.u_hrrr, snapshots.v_hrrr)];
       linestyle = :dash, label = "HRRR analysis")
axislegend(ax; position = :rb)
save(joinpath(FIGURES, "atmosphere_timeseries.png"), fig)
@info "Figures written" readdir(FIGURES)
