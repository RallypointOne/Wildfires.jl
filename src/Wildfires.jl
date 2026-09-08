module Wildfires

using Oceananigans
using Breeze

export ProjectedCRS, utm_zone, utm_epsg,
       to_grid, to_lonlat, to_grid_transform, from_grid_transform,
       raster_sampler, raster_topography, raster_time_series,
       fire_grid, FireModel, ignite!, advance!,
       spread_rate!, wind_adjustment, length_to_breadth, NormalProjection, HuygensEllipse, FuelMap,
       FuelClasses, FuelModel, FuelBed, spread_rate,
       polygon_coverage,
       HamadaModel, hamada_rates, ellipse_speed,
       BuildingClass, Structures, update_structures!, heat_release, heat_release!

include("georeference.jl")
include("fire_model.jl")
include("rothermel.jl")
include("hamada.jl")
include("polygons.jl")
include("structures.jl")
include("spread.jl")

end # module Wildfires
