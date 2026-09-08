# Shared setup for the Marshall Fire examples: data location, georeference,
# and the terrain-following atmosphere grid. Included by marshall_fire.jl and
# atmosphere.jl.

const DATA = joinpath(@__DIR__, "..", "..", "docs", "data", "marshall")
const T₀ = DateTime(2021, 12, 30, 18)            # ignition, UTC; model time is seconds after
const STOP = DateTime(2021, 12, 31, 2)           # last HRRR snapshot; spread had stopped by then

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
