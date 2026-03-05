#-----------------------------------------------------------------------------# Generate hero.gif for the Wildfires.jl docs landing page
#
# Usage:  julia --project=docs docs/generate_hero.jl
#
# Runs the Marshall Fire simulation and renders a 3D animation of fire
# spreading over terrain.  Outputs `docs/hero.gif`.

using Wildfires
using Wildfires.Rothermel
using Wildfires.Rothermel: NFFL_MODELS, nffl_model
using Wildfires.LevelSet
using Wildfires.LevelSet: xcoords, ycoords
using Wildfires.SpreadModels
using GLMakie
using Dates
using Rasters, ArchGDAL, GeoSurrogates
using Downloads

const DOCSDIR = @__DIR__
const DATADIR = joinpath(DOCSDIR, "data", "marshall")

#-----------------------------------------------------------------------------# Download data if missing
const DATA_URL = "https://github.com/RallypointOne/Wildfires.jl/releases/download/marshall-fire-data/marshall-fire-data.tar.gz"

if !isfile(joinpath(DATADIR, "slope.tif"))
    mkpath(DATADIR)
    tarball = joinpath(DATADIR, "marshall-fire-data.tar.gz")
    println("Downloading Marshall Fire data...")
    Downloads.download(DATA_URL, tarball)
    run(`tar xzf $tarball -C $DATADIR`)
    rm(tarball)
    println("  Extracted data to $DATADIR")
end

include(joinpath(DATADIR, "manifest.jl"))

#-----------------------------------------------------------------------------# Load LANDFIRE + HRRR data
slope_wrap     = RasterWrap(Raster(joinpath(DATADIR, "slope.tif")))
aspect_wrap    = RasterWrap(Raster(joinpath(DATADIR, "aspect.tif")))
elevation_wrap = RasterWrap(Raster(joinpath(DATADIR, "elevation.tif")))
fuel_wrap      = RasterWrap(Raster(joinpath(DATADIR, "fuel.tif")))

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

function wind_at(wf::WindField, lon, lat, dt::DateTime)
    times = wf.times
    dt <= first(times) && return _wind_snapshot(wf, 1, lon, lat)
    dt >= last(times) && return _wind_snapshot(wf, length(times), lon, lat)
    i = searchsortedlast(times, dt)
    alpha = Dates.value(dt - times[i]) / Dates.value(times[i + 1] - times[i])
    w1 = _wind_snapshot(wf, i, lon, lat)
    w2 = _wind_snapshot(wf, i + 1, lon, lat)
    (u = (1 - alpha) * w1.u + alpha * w2.u,
     v = (1 - alpha) * w1.v + alpha * w2.v,
     gust = (1 - alpha) * w1.gust + alpha * w2.gust)
end

function _wind_snapshot(wf::WindField, i, lon, lat)
    (u = predict(wf.u_wraps[i], (lon, lat)),
     v = predict(wf.v_wraps[i], (lon, lat)),
     gust = predict(wf.gust_wraps[i], (lon, lat)))
end

wind_field = load_wind_field(DATADIR, WIND_SUFFIXES, WIND_TIMES)

#-----------------------------------------------------------------------------# Custom components
const NON_FUEL_CODES = Set([91, 92, 93, 98, 99])

struct MarshallFuel <: AbstractFuel
    fuel_wrap::RasterWrap
    lon0::Float64
    lat0::Float64
end

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

#-----------------------------------------------------------------------------# Simulation
println("Setting up simulation...")
nx, ny = 290, 210
dx = 30.0
x0 = -(IGNITION_POINT.lon - EXTENT.X[1]) * M_PER_DEG_LON
y0 = -(IGNITION_POINT.lat - EXTENT.Y[1]) * M_PER_DEG_LAT

g = LevelSetGrid(nx, ny, dx=dx, x0=x0, y0=y0)
ignite!(g, 0.0, 0.0, 60.0)

fuel = MarshallFuel(fuel_wrap, IGNITION_POINT.lon, IGNITION_POINT.lat)
wind = MarshallWind(wind_field, DateTime(2021, 12, 30, 17),
    IGNITION_POINT.lon, IGNITION_POINT.lat, 0.31)
terrain = MarshallTerrain(slope_wrap, aspect_wrap,
    IGNITION_POINT.lon, IGNITION_POINT.lat)
moisture = UniformMoisture(FuelClasses(d1=0.03, d10=0.04, d100=0.05,
    herb=0.30, wood=0.50))

xs = collect(xcoords(g))
ys = collect(ycoords(g))
for j in eachindex(xs), i in eachindex(ys)
    if is_non_fuel(fuel, xs[j], ys[i])
        g.t_ignite[i, j] = NaN
    end
end

model = RothermelModel(fuel, wind, moisture, terrain, EllipticalBlending())

# Run simulation for 90 minutes, capturing a snapshot every minute
const SIM_MINUTES = 90
snapshots = Tuple{Float64, Matrix{Float64}}[(0.0, collect(g.φ))]

println("Running simulation (target: $SIM_MINUTES min)...")
est_steps = 10
for target_min in 1:SIM_MINUTES
    while g.t < target_min
        t_before = g.t
        simulate!(g, model, steps=est_steps)
        dt_actual = g.t - t_before
        if dt_actual > 0
            global est_steps = max(10, ceil(Int, 1.0 / dt_actual * est_steps))
        end
    end
    push!(snapshots, (g.t, collect(g.φ)))
    n_burned = count(<(0), g.φ)
    print("\r  t = $(round(g.t, digits=1)) min | burned = $n_burned/$(length(g.φ))")
end
println("\n  Captured $(length(snapshots)) frames (0 to $SIM_MINUTES min)")

#-----------------------------------------------------------------------------# Elevation matrix on simulation grid
println("\nBuilding elevation matrix...")
elev = [begin
    lon = IGNITION_POINT.lon + xs[j] / M_PER_DEG_LON
    lat = IGNITION_POINT.lat + ys[i] / M_PER_DEG_LAT
    predict(elevation_wrap, (lon, lat))
end for i in eachindex(ys), j in eachindex(xs)]

#-----------------------------------------------------------------------------# Record 3D hero GIF
using Wildfires.LevelSet: LevelSetGrid
using Makie

# Build a color matrix in-place: terrain colormap for unburned, solid red for burned
function _hero_colors!(colors, φ, elev, e_min, inv_e_range, terrain_cmap)
    n = length(terrain_cmap)
    for j in axes(φ, 2), i in axes(φ, 1)
        if φ[i, j] < 0
            colors[i, j] = Makie.RGBAf(0.8, 0.1, 0.1, 1.0)
        else
            frac = clamp((elev[i, j] - e_min) * inv_e_range, 0.0, 1.0)
            idx = clamp(round(Int, frac * (n - 1)) + 1, 1, n)
            colors[i, j] = terrain_cmap[idx]
        end
    end
    colors
end

outpath = joinpath(DOCSDIR, "hero.gif")
println("Recording hero.gif...")

# Transpose once: grid is ny×nx, Makie wants nx×ny
elev_t = collect(elev')

# Pre-compute elevation invariants
e_min, e_max = extrema(elev_t)
e_range = e_max - e_min
inv_e_range = e_range > 0 ? inv(e_range) : one(e_range)
terrain_cmap = Makie.to_colormap(:terrain)

# Pre-allocate buffers for transpose and colors
φ_buf = Matrix{Float64}(undef, size(elev_t))
colors_buf = Matrix{Makie.RGBAf}(undef, size(elev_t))

# Initialize with first frame
permutedims!(φ_buf, snapshots[1][2], (2, 1))
_hero_colors!(colors_buf, φ_buf, elev_t, e_min, inv_e_range, terrain_cmap)
colors_obs = Observable(copy(colors_buf))

fig = Figure(size=(700, 500))
ax = Axis3(fig[1, 1], xlabel="", ylabel="", zlabel="",
    azimuth=-0.3π, elevation=0.45π, protrusions=0)
surface!(ax, xs, ys, elev_t; color=colors_obs)

nframes = length(snapshots)
record(fig, outpath, eachindex(snapshots); framerate=15) do idx
    t, φ = snapshots[idx]
    permutedims!(φ_buf, φ, (2, 1))
    _hero_colors!(colors_buf, φ_buf, elev_t, e_min, inv_e_range, terrain_cmap)
    colors_obs[] = colors_buf
    ax.title = "Marshall Fire — t = $(round(t, digits=1)) min"
    print("\r  frame $idx/$nframes")
end
println("\nDone: $outpath ($(round(filesize(outpath) / 1024, digits=1)) KB)")
