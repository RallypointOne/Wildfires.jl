module Environment

using ..RateOfSpread: AbstractRateOfSpread, RothermelROS, McArthurGrasslandROS, McArthurForestROS,
    CanadianFBP_ROS, FuelClasses, rate_of_spread
using ..FireShape: AbstractFireShape, EllipticalShape, length_to_breadth

export NoWind, UniformWind
export UniformMoisture, GrasslandWeather, ForestWeather, FBPWeather
export FlatTerrain, UniformSlope
export FireEnvironment
export head_and_base

# ============================================================================ #
#                              Wind
# ============================================================================ #

"""No wind (speed=0)."""
struct NoWind end
(::NoWind)(t, x, y) = (0.0, 0.0)

"""
    UniformWind(; speed, direction=0.0)

Spatially and temporally constant wind field.

- `speed` — midflame wind speed [km/h]
- `direction` — wind direction [radians], meteorological convention (direction wind blows FROM)
"""
struct UniformWind{T}
    speed::T
    direction::T
end
UniformWind(; speed, direction=0.0) = UniformWind(promote(speed, direction)...)
(w::UniformWind)(t, x, y) = (w.speed, w.direction)

# ============================================================================ #
#                        Weather / Moisture
# ============================================================================ #

"""
    UniformMoisture(moisture::FuelClasses)

Spatially and temporally constant fuel moisture for the Rothermel model.

### Examples
```julia
UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0))
```
"""
struct UniformMoisture{T}
    moisture::FuelClasses{T}
end
(m::UniformMoisture)(t, x, y) = m.moisture

"""
    GrasslandWeather(; temperature, humidity, curing, fuel_load=4.5)

Uniform weather for the McArthur grassland fire model.

- `temperature` — air temperature [°C]
- `humidity` — relative humidity [%]
- `curing` — grass curing [%] (0=green, 100=fully cured)
- `fuel_load` — fuel load [t/ha]
"""
struct GrasslandWeather{T}
    temperature::T
    humidity::T
    curing::T
    fuel_load::T
end
GrasslandWeather(; temperature, humidity, curing, fuel_load=4.5) =
    GrasslandWeather(promote(temperature, humidity, curing, fuel_load)...)
(w::GrasslandWeather)(t, x, y) = (; temperature=w.temperature, humidity=w.humidity,
    curing=w.curing, fuel_load=w.fuel_load)

"""
    ForestWeather(; temperature, humidity, drought_factor, fuel_load=12.5)

Uniform weather for the McArthur forest fire model.

- `temperature` — air temperature [°C]
- `humidity` — relative humidity [%]
- `drought_factor` — drought factor [0–10]
- `fuel_load` — surface fuel load [t/ha]
"""
struct ForestWeather{T}
    temperature::T
    humidity::T
    drought_factor::T
    fuel_load::T
end
ForestWeather(; temperature, humidity, drought_factor, fuel_load=12.5) =
    ForestWeather(promote(temperature, humidity, drought_factor, fuel_load)...)
(w::ForestWeather)(t, x, y) = (; temperature=w.temperature, humidity=w.humidity,
    drought_factor=w.drought_factor, fuel_load=w.fuel_load)

"""
    FBPWeather(; ffmc, bui, curing=nothing)

Uniform weather for the Canadian FBP fire model.

- `ffmc` — Fine Fuel Moisture Code [0–101]
- `bui` — Buildup Index
- `curing` — grass curing [%] (only for O-1a/O-1b fuel types; `nothing` otherwise)
"""
struct FBPWeather{T, C}
    ffmc::T
    bui::T
    curing::C
end
FBPWeather(; ffmc, bui, curing=nothing) = FBPWeather(promote(ffmc, bui)..., curing)
(w::FBPWeather)(t, x, y) = (; ffmc=w.ffmc, bui=w.bui, curing=w.curing)

# ============================================================================ #
#                             Terrain
# ============================================================================ #

"""Flat terrain (zero slope everywhere)."""
struct FlatTerrain end
(::FlatTerrain)(t, x, y) = (zero(x), zero(x))

"""
    UniformSlope(; slope, aspect=0.0)

Spatially constant terrain slope.

- `slope` — terrain slope as rise/run [fraction]
- `aspect` — slope aspect [radians]
"""
struct UniformSlope{T}
    slope::T
    aspect::T
end
UniformSlope(; slope, aspect=0.0) = UniformSlope(promote(slope, aspect)...)
(s::UniformSlope)(t, x, y) = (s.slope, s.aspect)

# ============================================================================ #
#                         FireEnvironment
# ============================================================================ #

"""
    FireEnvironment(wind, weather, terrain)

Container for spatially varying environmental fields. Each field is a callable
`(t, x, y) -> data` whose return type depends on the ROS model in use.

### Examples
```julia
# Rothermel environment
env = FireEnvironment(
    UniformWind(speed=8.0),
    UniformMoisture(FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)),
    FlatTerrain(),
)

# McArthur grassland environment
env = FireEnvironment(
    UniformWind(speed=20.0),
    GrasslandWeather(temperature=35.0, humidity=15.0, curing=90.0),
    FlatTerrain(),
)
```
"""
struct FireEnvironment{W, X, T}
    wind::W
    weather::X
    terrain::T
end

# ============================================================================ #
#                          head_and_base
# ============================================================================ #

"""
    head_and_base(ros, env, shape, t, x, y)

Compute head-fire and backing rates, push direction, and length-to-breadth ratio
for a given ROS model, environment, and fire shape at position `(x, y)` and time `t`.

Returns a `NamedTuple` `(; R_head, R_base, push_x, push_y, LB)` where:
- `R_head` — head-fire rate of spread [m/min]
- `R_base` — backing rate of spread [m/min]
- `push_x, push_y` — unit vector of dominant spread direction
- `LB` — fire length-to-breadth ratio
"""
function head_and_base end

#-----------------------------------------------------------------------------# Rothermel
function head_and_base(ros::RothermelROS, env::FireEnvironment, shape::AbstractFireShape, t, x, y)
    speed, wind_dir = env.wind(t, x, y)
    moist = env.weather(t, x, y)
    slope_val, aspect = env.terrain(t, x, y)

    R_head = rate_of_spread(ros; moisture=moist, wind=speed, slope=slope_val)
    R_base = rate_of_spread(ros; moisture=moist, wind=0.0, slope=0.0)

    # Push direction from combined wind + slope effects
    R_w = rate_of_spread(ros; moisture=moist, wind=speed, slope=0.0)
    R_s = rate_of_spread(ros; moisture=moist, wind=0.0, slope=slope_val)
    w_wind = R_w - R_base
    w_slope = R_s - R_base

    push_x = w_wind * (-cos(wind_dir)) + w_slope * (-cos(aspect))
    push_y = w_wind * (-sin(wind_dir)) + w_slope * (-sin(aspect))
    pmag = hypot(push_x, push_y)
    if pmag > 0
        push_x /= pmag
        push_y /= pmag
    end

    LB = _compute_lb(shape, ros, moist, R_head, R_base)

    return (; R_head, R_base, push_x, push_y, LB)
end

#-----------------------------------------------------------------------------# McArthur Grassland
function head_and_base(ros::McArthurGrasslandROS, env::FireEnvironment, shape::AbstractFireShape, t, x, y)
    speed, wind_dir = env.wind(t, x, y)
    wx = env.weather(t, x, y)

    R_head = rate_of_spread(ros; wind=speed, temperature=wx.temperature,
        humidity=wx.humidity, curing=wx.curing, fuel_load=wx.fuel_load)
    R_base = rate_of_spread(ros; wind=0.0, temperature=wx.temperature,
        humidity=wx.humidity, curing=wx.curing, fuel_load=wx.fuel_load)

    push_x = speed > 0 ? -cos(wind_dir) : zero(speed)
    push_y = speed > 0 ? -sin(wind_dir) : zero(speed)

    LB = _compute_lb_from_wind(shape, speed)

    return (; R_head, R_base, push_x, push_y, LB)
end

#-----------------------------------------------------------------------------# McArthur Forest
function head_and_base(ros::McArthurForestROS, env::FireEnvironment, shape::AbstractFireShape, t, x, y)
    speed, wind_dir = env.wind(t, x, y)
    wx = env.weather(t, x, y)

    R_head = rate_of_spread(ros; wind=speed, temperature=wx.temperature,
        humidity=wx.humidity, drought_factor=wx.drought_factor, fuel_load=wx.fuel_load)
    R_base = rate_of_spread(ros; wind=0.0, temperature=wx.temperature,
        humidity=wx.humidity, drought_factor=wx.drought_factor, fuel_load=wx.fuel_load)

    push_x = speed > 0 ? -cos(wind_dir) : zero(speed)
    push_y = speed > 0 ? -sin(wind_dir) : zero(speed)

    LB = _compute_lb_from_wind(shape, speed)

    return (; R_head, R_base, push_x, push_y, LB)
end

#-----------------------------------------------------------------------------# Canadian FBP
function head_and_base(ros::CanadianFBP_ROS, env::FireEnvironment, shape::AbstractFireShape, t, x, y)
    speed, wind_dir = env.wind(t, x, y)
    wx = env.weather(t, x, y)

    R_head = rate_of_spread(ros; ffmc=wx.ffmc, wind=speed, bui=wx.bui, curing=wx.curing)
    R_base = rate_of_spread(ros; ffmc=wx.ffmc, wind=0.0, bui=wx.bui, curing=wx.curing)

    push_x = speed > 0 ? -cos(wind_dir) : zero(speed)
    push_y = speed > 0 ? -sin(wind_dir) : zero(speed)

    LB = _compute_lb_from_wind(shape, speed)

    return (; R_head, R_base, push_x, push_y, LB)
end

# ============================================================================ #
#                   LB Computation Helpers
# ============================================================================ #

# Default: no elliptical shape → LB = 1
_compute_lb(::AbstractFireShape, ros, moist, R_head, R_base) = one(R_head)

function _compute_lb(shape::EllipticalShape, ros::RothermelROS, moist, R_head, R_base)
    U_eff_kmh = _effective_wind_speed(ros, moist, R_head, R_base)
    U_eff_ms = U_eff_kmh / 3.6
    length_to_breadth(U_eff_ms; formula=shape.formula)
end

_compute_lb_from_wind(::AbstractFireShape, speed) = one(speed)

function _compute_lb_from_wind(shape::EllipticalShape, speed)
    U_ms = speed / 3.6  # km/h → m/s
    length_to_breadth(U_ms; formula=shape.formula)
end

#-----------------------------------------------------------------------------# _effective_wind_speed (Rothermel bisection)
"""
Find the wind speed [km/h] that produces `R_target` from the Rothermel model
with zero slope. Used to convert the combined wind+slope ROS into an effective
wind speed for LB computation.
"""
function _effective_wind_speed(ros::RothermelROS, moist, R_target, R_base)
    R_target <= R_base && return 0.0
    lo, hi = 0.0, 50.0
    while rate_of_spread(ros; moisture=moist, wind=hi, slope=0.0) < R_target
        hi *= 2
    end
    for _ in 1:20
        mid = (lo + hi) / 2
        R_mid = rate_of_spread(ros; moisture=moist, wind=mid, slope=0.0)
        if R_mid < R_target
            lo = mid
        else
            hi = mid
        end
        (hi - lo) < 0.1 && break
    end
    return (lo + hi) / 2
end

end # module
