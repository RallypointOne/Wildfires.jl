# Marshall Fire hindcast
#
# Boulder County, Colorado, 30 December 2021. The fire ignited near Marshall
# Mesa around 18:00 UTC (11:00 MST) during a downslope windstorm with west
# winds of 20–30 m/s and gusts near 50 m/s, ran about 10 km east through
# Superior and Louisville, and stopped overnight under snowfall. The final
# WFIGS perimeter (`perimeter.geojson`, GISAcres) is 6,026 acres = 24.4 km²;
# the county counted 1,084 structures destroyed.
#
# Inputs, all in docs/data/marshall (see the download and prepare scripts):
#   elevation.tif, slope.tif, aspect.tif   LANDFIRE 2020, WGS84, ~30 m
#   fuel.tif                               LANDFIRE FBFM13 fuel model codes
#   wind_{u,v,gust}_HHMM.tif               HRRR 10 m wind, 15 min, 17:00–02:00 UTC
#   perimeter.geojson                      WFIGS final perimeter, WGS84
#   buildings.geojson                      OpenStreetMap footprints
#
# Run with:  julia --project=examples/marshall examples/marshall/marshall_fire.jl
#
# Sections marked BLOCKED name a package feature the hindcast still lacks.

using Wildfires
using Oceananigans
using Oceananigans.Grids: xspacings, yspacings, xnodes, ynodes, Center
using Oceananigans.OutputReaders: Time
using Breeze.TerrainFollowingDiscretization
using Rasters, ArchGDAL
using GeoJSON
using Dates, Printf

const ACRE = 4046.86
include("domain.jl")

#-----------------------------------------------------------------------------# Fire grid and model
# Refinement 8 gives 25 m fire cells, close to the 30 m LANDFIRE pixels.
grid = fire_grid(atmosphere_grid; refinement = 8)
model = FireModel(grid)
fire_field() = Field{Center, Center, Nothing}(grid)

#-----------------------------------------------------------------------------# Fuel
# FBFM13 codes, nearest-neighbour, one Rothermel bed per code on the device.
# LANDFIRE classes developed land (a third of this domain) as 91, not
# burnable, which would stop the vegetation front at every street; lawns and
# ornamental plantings, cured in December, carry fire between houses, so 91
# is treated as short grass (fuel model 1), as WRF-Fire WUI studies do.
# Snow, agriculture, water, and barren (93–99) stay non-burnable. Buildings
# themselves spread through `structures`.
fuel = FuelMap(grid, raster_sampler(Raster(joinpath(DATA, "fuel.tif")), crs; method = :near), Wildfires.NFFL;
               remap = (91 => 1,))

# Fully cured grass after a record-dry autumn: 6% dead, 30% live.
moisture = FuelClasses(0.06, 0.30)

#-----------------------------------------------------------------------------# Slope
# LANDFIRE slope (degrees) and aspect (downslope azimuth, degrees clockwise
# from north, -1 flat) give the terrain gradient directly; ∇h points upslope.
# Nearest-neighbour for both, since aspect wraps at 0/360.
slope_deg = raster_sampler(Raster(joinpath(DATA, "slope.tif")), crs; method = :near)
aspect_deg = raster_sampler(Raster(joinpath(DATA, "aspect.tif")), crs; method = :near, fill_value = -1)
∂x_h = fire_field()
∂y_h = fire_field()
set!(∂x_h, (x, y) -> -tand(slope_deg(x, y)) * sind(aspect_deg(x, y)))
set!(∂y_h, (x, y) -> -tand(slope_deg(x, y)) * cosd(aspect_deg(x, y)))

#-----------------------------------------------------------------------------# Structures
# OpenStreetMap footprints over the domain. No construction attributes are
# available, so every building is the default class; the `building` tag
# could map to fire-resistant classes once damage inspection data says
# which types survived.
buildings = GeoJSON.read(joinpath(DATA, "buildings.geojson"))
structures = Structures(buildings, crs, grid)
println(structures)

#-----------------------------------------------------------------------------# Wind
# HRRR 10 m wind, bilinear from the 3 km grid, as a time series interpolated
# linearly between 15 min snapshots and read at the model clock. Passed as the
# open wind; `spread_rate!` applies each fuel model's adjustment factor for
# vegetation and uses the open wind for Hamada.
#
# BLOCKED: gusts (wind_gust_*.tif, up to 40 m/s here) are not used; the
# sustained wind under-represents this storm.
stamps = T₀:Minute(15):STOP
times = [Dates.value(Second(t - T₀)) for t in stamps]
wind(component) = raster_time_series([Raster(joinpath(DATA, "wind_$(component)_$(Dates.format(t, "HHMM")).tif"))
                                      for t in stamps], times, grid, crs)
u = wind("u")
v = wind("v")

#-----------------------------------------------------------------------------# Hindcast
# The ignition pixel and its neighbours are urban (code 91) in LANDFIRE, so a
# 100 m circle seeds the grass around the property; at 50 m the fire never
# leaves it.
ignite!(model, x_ign, y_ign; radius = 100.0)

speed = fire_field()
Δx_fire = first(xspacings(grid, Center()))
Δy_fire = first(yspacings(grid, Center()))
stop_time = last(times)
burned_area() = count(isfinite, interior(model.t_ignition)) * Δx_fire * Δy_fire
i_ign = argmin(abs.(xnodes(grid, Center()) .- x_ign))
j_ign = argmin(abs.(ynodes(grid, Center()) .- y_ign))
wind_at_ignition(t) = hypot(interior(u[Time(t)])[i_ign, j_ign, 1], interior(v[Time(t)])[i_ign, j_ign, 1])

# Every five minutes: time [s], burned area [m²], structures ignited, total
# structure heat release [W], and the 10 m wind at the ignition point [m/s].
history = (t = Float64[], area = Float64[], structures = Int[], hrr = Float64[], wind = Float64[])
function record!(history, t)
    push!(history.t, t)
    push!(history.area, burned_area())
    push!(history.structures, count(isfinite, structures.t_ignition))
    push!(history.hrr, sum(heat_release(structures, t)))
    push!(history.wind, wind_at_ignition(t))
end

next_record = 0.0
while model.clock.time < stop_time
    global next_record
    if model.clock.time >= next_record
        t = model.clock.time
        record!(history, t)
        if t % 3600 == 0
            @printf("%s UTC: burned %5.1f km², %4d structures ignited, wind at ignition %4.1f m/s\n",
                    Dates.format(T₀ + Second(round(Int, t)), "HH:MM"), burned_area() / 1e6,
                    count(isfinite, structures.t_ignition), wind_at_ignition(t))
        end
        next_record += 300
    end
    spread_rate!(speed, model, fuel, moisture; wind = (u, v), slope = (∂x_h, ∂y_h), structures)
    Δt = 0.5 / (maximum(interior(speed)) * (1 / Δx_fire + 1 / Δy_fire))   # half the Godunov/Heun limit
    advance!(model, min(Δt, next_record - model.clock.time, stop_time - model.clock.time); speed)
    update_structures!(structures, model)
end
record!(history, model.clock.time)

#-----------------------------------------------------------------------------# Against the observed perimeter
observed = polygon_coverage(GeoJSON.read(joinpath(DATA, "perimeter.geojson")), crs, grid)
burned = isfinite.(interior(model.t_ignition))
inside = interior(observed) .> 0.5
cell = Δx_fire * Δy_fire
hit = count(burned .& inside) * cell
miss = count(.!burned .& inside) * cell
false_alarm = count(burned .& .!inside) * cell

@printf("\nafter %.0f h, %s UTC\n", model.clock.time / 3600, Dates.format(STOP, "HH:MM"))
@printf("burned area:      %5.1f km² (%5.0f acres)\n", burned_area() / 1e6, burned_area() / ACRE)
@printf("observed:         %5.1f km² (%5.0f acres)\n", count(inside) * cell / 1e6, count(inside) * cell / ACRE)
@printf("hit / miss / false alarm: %.1f / %.1f / %.1f km², critical success index %.2f\n",
        hit / 1e6, miss / 1e6, false_alarm / 1e6, hit / (hit + miss + false_alarm))
@printf("structures ignited: %d of %d; observed destroyed: 1,084\n",
        count(isfinite, structures.t_ignition), length(structures))
@printf("peak structure heat release: %.2f GW\n", maximum(history.hrr) / 1e9)

#-----------------------------------------------------------------------------# Figures
# Inputs, arrival time against the observed perimeter, time series, and an
# animation of the front, written to figures/.
include("figures.jl")
