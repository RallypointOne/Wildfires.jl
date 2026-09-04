module Wildfires

using Oceananigans
using Breeze

export ProjectedCRS, utm_zone, utm_epsg,
       to_grid, to_lonlat, to_grid_transform, from_grid_transform,
       raster_topography,
       fire_grid, FireModel, ignite!, advance!,
       FuelClasses, FuelModel, FuelBed, spread_rate

include("georeference.jl")
include("fire_model.jl")
include("rothermel.jl")

end # module Wildfires
