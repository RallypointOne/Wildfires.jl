using Wildfires
using Wildfires.Rothermel
using Wildfires.Rothermel: NFFL_MODELS, nffl_model
using Wildfires.LevelSet
using Wildfires.SpreadModels
using Dates
using Rasters, ArchGDAL, GeoSurrogates

const DATADIR = joinpath(@__DIR__, "..", "docs", "data", "marshall")
include(joinpath(DATADIR, "manifest.jl"))

# Load terrain and fuel
slope_wrap = RasterWrap(Raster(joinpath(DATADIR, "slope.tif")))
aspect_wrap = RasterWrap(Raster(joinpath(DATADIR, "aspect.tif")))
fuel_wrap = RasterWrap(Raster(joinpath(DATADIR, "fuel.tif")))

# Load wind (15-min resolution with gust)
struct WindField
    times::Vector{DateTime}
    u_wraps::Vector{<:RasterWrap}
    v_wraps::Vector{<:RasterWrap}
    gust_wraps::Vector{<:RasterWrap}
end

function load_wind_field(datadir, suffixes, times)
    u_wraps    = [RasterWrap(Raster(joinpath(datadir, "wind_u_$s.tif"))) for s in suffixes]
    v_wraps    = [RasterWrap(Raster(joinpath(datadir, "wind_v_$s.tif"))) for s in suffixes]
    gust_wraps = [RasterWrap(Raster(joinpath(datadir, "wind_gust_$s.tif"))) for s in suffixes]
    WindField(times, u_wraps, v_wraps, gust_wraps)
end

function _wind_snapshot(wf::WindField, i, lon, lat)
    (u    = predict(wf.u_wraps[i], (lon, lat)),
     v    = predict(wf.v_wraps[i], (lon, lat)),
     gust = predict(wf.gust_wraps[i], (lon, lat)))
end

function wind_at(wf::WindField, lon, lat, dt::DateTime)
    times = wf.times
    if dt <= first(times)
        return _wind_snapshot(wf, 1, lon, lat)
    elseif dt >= last(times)
        return _wind_snapshot(wf, length(times), lon, lat)
    end
    i = searchsortedlast(times, dt)
    t1, t2 = times[i], times[i + 1]
    alpha = Dates.value(dt - t1) / Dates.value(t2 - t1)
    w1 = _wind_snapshot(wf, i, lon, lat)
    w2 = _wind_snapshot(wf, i + 1, lon, lat)
    return (u    = (1 - alpha) * w1.u    + alpha * w2.u,
            v    = (1 - alpha) * w1.v    + alpha * w2.v,
            gust = (1 - alpha) * w1.gust + alpha * w2.gust)
end

wind_field = load_wind_field(DATADIR, WIND_SUFFIXES, WIND_TIMES)

# Print wind at ignition point at 17:00
wnd = wind_at(wind_field, IGNITION_POINT.lon, IGNITION_POINT.lat, DateTime(2021, 12, 30, 17))
println("Wind at ignition (17:00 UTC):")
println("  u (east) = $(round(wnd.u, digits=1)) m/s")
println("  v (north) = $(round(wnd.v, digits=1)) m/s")
println("  gust = $(round(wnd.gust, digits=1)) m/s ($(round(wnd.gust * 2.237, digits=1)) mph)")
println("  mean speed = $(round(hypot(wnd.u, wnd.v), digits=1)) m/s")
println("  FROM direction (math) = $(round(rad2deg(atan(-wnd.v, -wnd.u)), digits=1))°")

# Custom fuel component
struct MarshallFuel <: AbstractFuel
    fuel_wrap::RasterWrap
    lon0::Float64
    lat0::Float64
end

const NON_FUEL_CODES = Set([91, 92, 93, 98, 99])

function (f::MarshallFuel)(t, x, y)
    lon = f.lon0 + x / M_PER_DEG_LON
    lat = f.lat0 + y / M_PER_DEG_LAT
    code = round(Int, predict(f.fuel_wrap, (lon, lat)))
    model = nffl_model(code)
    return model !== nothing ? model : SHORT_GRASS
end

function is_non_fuel(f::MarshallFuel, x, y)
    lon = f.lon0 + x / M_PER_DEG_LON
    lat = f.lat0 + y / M_PER_DEG_LAT
    code = round(Int, predict(f.fuel_wrap, (lon, lat)))
    return code ∈ NON_FUEL_CODES
end

# Custom components with fixes
struct MarshallWind <: AbstractWind
    field::WindField
    start_time::DateTime
    lon0::Float64
    lat0::Float64
    adjustment::Float64
end

function (w::MarshallWind)(t, x, y)
    lon = w.lon0 + x / M_PER_DEG_LON
    lat = w.lat0 + y / M_PER_DEG_LAT
    dt = w.start_time + Second(round(Int, t * 60))
    wnd = wind_at(w.field, lon, lat, dt)
    speed_kmh = hypot(wnd.u, wnd.v) * 3.6 * w.adjustment
    direction = atan(-wnd.v, -wnd.u)
    return (speed_kmh, direction)
end

struct MarshallTerrain <: AbstractTerrain
    slp_wrap::RasterWrap
    asp_wrap::RasterWrap
    lon0::Float64
    lat0::Float64
end

function (t::MarshallTerrain)(_, x, y)
    lon = t.lon0 + x / M_PER_DEG_LON
    lat = t.lat0 + y / M_PER_DEG_LAT
    slp_deg = predict(t.slp_wrap, (lon, lat))
    slope = tan(deg2rad(clamp(slp_deg, 0.0, 80.0)))
    aspect = π/2 - deg2rad(predict(t.asp_wrap, (lon, lat)))
    return (slope, aspect)
end

# Grid setup
nx, ny = 290, 210
dx = 30.0
x0 = -(IGNITION_POINT.lon - EXTENT.X[1]) * M_PER_DEG_LON
y0 = -(IGNITION_POINT.lat - EXTENT.Y[1]) * M_PER_DEG_LAT

g = LevelSetGrid(nx, ny, dx=dx, x0=x0, y0=y0)
ignite!(g, 0.0, 0.0, 60.0)

fuel = MarshallFuel(fuel_wrap, IGNITION_POINT.lon, IGNITION_POINT.lat)
wind = MarshallWind(wind_field, DateTime(2021, 12, 30, 17), IGNITION_POINT.lon, IGNITION_POINT.lat, 0.31)
terrain = MarshallTerrain(slope_wrap, aspect_wrap, IGNITION_POINT.lon, IGNITION_POINT.lat)
moisture = UniformMoisture(FuelClasses(d1=0.03, d10=0.04, d100=0.05, herb=0.30, wood=0.50))

# Mark non-fuel cells as unburnable
xs = collect(xcoords(g))
ys = collect(ycoords(g))
for j in eachindex(xs), i in eachindex(ys)
    if is_non_fuel(fuel, xs[j], ys[i])
        g.t_ignite[i, j] = NaN
    end
end

model = RothermelModel(fuel, wind, moisture, terrain, EllipticalBlending())

# Quick test: spread rate at ignition point
rate = model(0.0, 0.0, 0.0)
println("\nSpread rate at ignition: $(round(rate, digits=2)) m/min")

# Also test directional speed in wind direction
spd, dir = wind(0.0, 0.0, 0.0)
println("Wind at ignition: speed=$(round(spd, digits=1)) km/h, from_dir=$(round(rad2deg(dir), digits=1))°")

# Run simulation
println("\nRunning simulation...")
simulate!(g, model, steps=1000, progress=true)
println("\nSimulation complete: t = $(round(g.t, digits=1)) min, burned = $(count(<(0), g.φ)) cells")
println("Burned area: $(round(burn_area(g) / 1e6, digits=2)) km²")
