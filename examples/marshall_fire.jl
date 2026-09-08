# Marshall Fire hindcast
#
# Boulder County, Colorado, 30 December 2021. The fire ignited near Marshall
# Mesa around 18:00 UTC (11:00 MST) during a downslope windstorm with west
# winds of 20–30 m/s and gusts near 50 m/s, ran about 10 km east through
# Superior and Louisville, and stopped overnight under snowfall. The final
# WFIGS perimeter (`perimeter.geojson`, GISAcres) is 6,026 acres = 24.4 km².
#
# Inputs, all in docs/data/marshall (see prepare_data.jl and download_hrrr.jl):
#   elevation.tif, slope.tif, aspect.tif   LANDFIRE 2020, WGS84, ~30 m
#   fuel.tif                               LANDFIRE FBFM13 fuel model codes
#   wind_{u,v,gust}_HHMM.tif               HRRR 10 m wind, 15 min, 17:00–19:00 UTC
#   perimeter.geojson                      WFIGS final perimeter, WGS84
#   buildings.geojson                      OpenStreetMap footprints (download_buildings.jl)
#
# Run with:  julia --project=examples examples/marshall_fire.jl
#
# Status: the parts that run use only implemented package features. Sections
# marked BLOCKED name the package feature the real hindcast needs and use a
# stated placeholder so the script executes end to end.

using Wildfires
using Oceananigans
using Oceananigans.Grids: xspacings, yspacings, Center
using Breeze.TerrainFollowingDiscretization
using Rasters, ArchGDAL
using GeoJSON
using Printf

const DATA = joinpath(@__DIR__, "..", "docs", "data", "marshall")
const OBSERVED_AREA = 6025.7 * 4046.86   # m², GISAcres in perimeter.geojson

#-----------------------------------------------------------------------------# Georeference
# UTM 13N, origin at the centre of the LANDFIRE tiles. The observed perimeter
# spans lon -105.233 to -105.131 and lat 39.929 to 39.986, which is inside the
# ±5 km × ±3.6 km domain below.
crs = ProjectedCRS(-105.182, 39.9575)

# Investigators placed the first ignition at 5325 Eldorado Springs Drive; the
# second, on the Xcel line near Highway 93, followed within the hour.
ignition_lonlat = (-105.231, 39.955)
x_ign, y_ign = to_grid(crs, ignition_lonlat...)

#-----------------------------------------------------------------------------# Atmosphere grid
# 200 m horizontal, terrain-following vertical. `z` is absolute altitude, so an
# 8 km top over 1.6–1.8 km terrain leaves ~6 km of atmosphere. TwoLevelDecay
# (SLEVE) rather than LinearDecay so LANDFIRE-scale terrain does not imprint
# on coordinate surfaces aloft.
Nx, Ny, Nz = 50, 36, 40
Δx = 200.0
z_top = 8000.0
z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, z_top, length = Nz + 1));
                                                 formulation = TwoLevelDecay(large_scale_height = z_top / 2,
                                                                             small_scale_height = z_top / 8))
atmosphere_grid = RectilinearGrid(size = (Nx, Ny, Nz), halo = (5, 5, 5),
                                  x = (-Nx * Δx / 2, Nx * Δx / 2),
                                  y = (-Ny * Δx / 2, Ny * Δx / 2),
                                  z = z_faces,
                                  topology = (Bounded, Bounded, Bounded))

dem = Raster(joinpath(DATA, "elevation.tif"))
materialize_terrain!(atmosphere_grid, raster_topography(dem, crs))

#-----------------------------------------------------------------------------# Fire grid and model
# Refinement 8 gives 25 m fire cells, close to the 30 m LANDFIRE pixels.
grid = fire_grid(atmosphere_grid; refinement = 8)
model = FireModel(grid)
fire_field() = Field{Center, Center, Nothing}(grid)

#-----------------------------------------------------------------------------# Fuel
# FBFM13 codes, nearest-neighbour. Burnable models are 1–13; 91–99 are urban,
# snow, agriculture, water, and barren, and 0 is off the raster. The mask
# zeroes vegetation spread there; urban cells spread through `structures`.
fuel_code = raster_sampler(Raster(joinpath(DATA, "fuel.tif")), crs; method = :near)
burnable = fire_field()
set!(burnable, (x, y) -> 1 <= fuel_code(x, y) <= 13)

# BLOCKED: per-cell fuel. `spread_rate!` takes one `FuelBed` for the whole
# domain, so every burnable cell is fuel model 2 (timber grass, 44% of pixels;
# most of the rest that burns is 8, closed timber litter). A device-side table
# of beds indexed by `fuel_code` is the fix.
bed = FuelBed(Wildfires.NFFL.TIMBER_GRASS)

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
# OpenStreetMap footprints: 12,709 polygons over the domain. No construction
# attributes are available, so every building is the default class; the
# `building` tag could map to fire-resistant classes once damage inspection
# data says which types survived.
buildings = GeoJSON.read(joinpath(DATA, "buildings.geojson"))
structures = Structures(buildings, crs, grid)
println(structures)

#-----------------------------------------------------------------------------# Wind
# HRRR 10 m wind, bilinear from the 3 km grid, held constant over each 15 min
# snapshot, passed as the open wind; `spread_rate!` applies the fuel model's
# wind adjustment factor for vegetation and uses the open wind for Hamada.
#
# BLOCKED: a wind time series on the fire grid belongs in the package. Gusts
# (wind_gust_*.tif, up to 40 m/s here) are not used; the sustained wind
# under-represents this storm.
snapshots = ["1800", "1815", "1830", "1845", "1900"]
u = fire_field()
v = fire_field()
function set_wind!(u, v, stamp)
    for (field, component) in ((u, "u"), (v, "v"))
        set!(field, raster_sampler(Raster(joinpath(DATA, "wind_$(component)_$stamp.tif")), crs))
    end
end

#-----------------------------------------------------------------------------# Hindcast
# t = 0 is 18:00 UTC. HRRR snapshots end at 19:00 UTC, so the run stops there;
# the fire kept growing until roughly 02:00 UTC on the 31st. Extending
# download_hrrr.jl's TIMESTEPS covers the rest. The ignition pixel and its
# neighbours are urban (code 91) in LANDFIRE, so a 100 m circle seeds the
# grass around the property; at 50 m the fire never leaves it.
ignite!(model, x_ign, y_ign; radius = 100.0)

speed = fire_field()
urban = fire_field()
wet = FuelClasses(1.0, 1.0)     # above extinction: the urban call carries no vegetation term
Δx_fire = first(xspacings(grid, Center()))
Δy_fire = first(yspacings(grid, Center()))
stop_time = 3600.0
snapshot = 0
while model.clock.time < stop_time
    global snapshot
    k = min(1 + floor(Int, model.clock.time / 900), length(snapshots))
    if k != snapshot
        set_wind!(u, v, snapshots[k])
        snapshot = k
        @printf("%s UTC: 10 m wind %.1f–%.1f m/s, %d structures ignited\n", snapshots[k],
                extrema(hypot.(interior(u), interior(v)))..., count(isfinite, structures.t_ignition))
    end
    # Vegetation spread only on burnable fuel; the mask applies to the urban
    # term as well, so it is recomputed on top afterwards.
    spread_rate!(speed, model, bed, moisture; wind = (u, v), slope = (∂x_h, ∂y_h))
    speed .*= burnable
    spread_rate!(urban, model, bed, wet; wind = (u, v), structures)
    interior(speed) .= max.(interior(speed), interior(urban))
    Δt = 0.5 / (maximum(interior(speed)) * (1 / Δx_fire + 1 / Δy_fire))   # half the Godunov/Heun limit
    advance!(model, min(Δt, stop_time - model.clock.time); speed)
    update_structures!(structures, model)
end

#-----------------------------------------------------------------------------# Diagnostics
burned_cells = count(isfinite, interior(model.t_ignition))
burned_area = burned_cells * Δx_fire * Δy_fire
@printf("head-fire rate at 19:00 UTC: %.2f m/s\n", maximum(interior(speed)))
@printf("burned area after %.0f s: %.1f km² (%.0f acres)\n",
        model.clock.time, burned_area / 1e6, burned_area / 4046.86)
@printf("observed final perimeter (~8 h): %.1f km² (%.0f acres)\n",
        OBSERVED_AREA / 1e6, OBSERVED_AREA / 4046.86)
@printf("structures ignited: %d of %d; observed destroyed (whole fire): 1,084\n",
        count(isfinite, structures.t_ignition), length(structures))
@printf("structure heat release at 19:00 UTC: %.2f GW\n", sum(heat_release(structures, model.clock.time)) / 1e9)

# BLOCKED: comparison against the observed perimeter. Needs the GeoJSON polygon
# rasterized onto the fire grid (also the polygon-ignition and
# reinitialize-from-observation path), then burned-area overlap statistics.
