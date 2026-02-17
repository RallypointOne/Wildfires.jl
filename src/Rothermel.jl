module Rothermel

export FuelClasses, Rothermel, rate_of_spread
export SHORT_GRASS, TIMBER_GRASS, TALL_GRASS, CHAPARRAL, BRUSH, DORMANT_BRUSH,
    SOUTHERN_ROUGH, CLOSED_TIMBER_LITTER, HARDWOOD_LITTER, TIMBER_UNDERSTORY,
    LIGHT_SLASH, MEDIUM_SLASH, HEAVY_SLASH

#=
    Rothermel (1972) Surface Fire Spread Model

    References:
    - Rothermel, R.C. (1972). A Mathematical Model for Predicting Fire Spread in
      Wildland Fuels. Res. Paper INT-115, USDA Forest Service.
    - Anderson, H.E. (1982). Aids to Determining Fuel Models for Estimating Fire
      Behavior. Gen. Tech. Rep. INT-122, USDA Forest Service.
    - Andrews, P.L. (2018). The Rothermel Surface Fire Spread Model and Associated
      Developments. Gen. Tech. Rep. RMRS-GTR-371, USDA Forest Service.
=#

#-----------------------------------------------------------------------------# FuelClasses
"""
    FuelClasses{T}(; d1, d10, d100, herb, wood)

Values for the five Rothermel fuel size classes.

# Fields
- `d1::T`   - 1-hr dead fuel (< 0.25 in diameter)
- `d10::T`  - 10-hr dead fuel (0.25–1.0 in)
- `d100::T` - 100-hr dead fuel (1.0–3.0 in)
- `herb::T` - Live herbaceous
- `wood::T` - Live woody
"""
Base.@kwdef struct FuelClasses{T}
    d1::T
    d10::T
    d100::T
    herb::T
    wood::T
end

FuelClasses(a, b, c, d, e) = FuelClasses(promote(a, b, c, d, e)...)

function Base.show(io::IO, fc::FuelClasses{T}) where {T}
    print(io, "FuelClasses{$T}(d1=$(fc.d1), d10=$(fc.d10), d100=$(fc.d100), herb=$(fc.herb), wood=$(fc.wood))")
end

Base.eltype(::Type{FuelClasses{T}}) where {T} = T
Base.map(f, a::FuelClasses) = FuelClasses(f(a.d1), f(a.d10), f(a.d100), f(a.herb), f(a.wood))
Base.map(f, a::FuelClasses, b::FuelClasses) = FuelClasses(f(a.d1, b.d1), f(a.d10, b.d10), f(a.d100, b.d100), f(a.herb, b.herb), f(a.wood, b.wood))
Base.sum(fc::FuelClasses) = fc.d1 + fc.d10 + fc.d100 + fc.herb + fc.wood
Base.sum(f::Function, fc::FuelClasses) = f(fc.d1) + f(fc.d10) + f(fc.d100) + f(fc.herb) + f(fc.wood)

#-----------------------------------------------------------------------------# Rothermel
"""
    Rothermel{T}(; name, w, σ, h, δ, Mx)

Fuel model for the Rothermel (1972) surface fire spread model.

Parameterized by numeric type `T` (e.g. `Float64`, `Float32`, Unitful quantities).

# Fields (US customary units, matching original publications)
- `name::String` - Description
- `w::FuelClasses{T}` - Fuel loading [tons/acre]
- `σ::FuelClasses{T}` - Surface-area-to-volume ratio [1/ft]
- `h::FuelClasses{T}` - Heat content [BTU/lb]
- `δ::T` - Fuel bed depth [ft]
- `Mx::T` - Dead fuel moisture of extinction [fraction]

See [`NFFL`](@ref) for the 13 standard fuel models from Anderson (1982).
"""
Base.@kwdef struct Rothermel{T}
    name::String = "Custom"
    w::FuelClasses{T}
    σ::FuelClasses{T}
    h::FuelClasses{T}
    δ::T
    Mx::T
end

function Rothermel(name::AbstractString, w, σ, h, δ, Mx)
    T = promote_type(eltype(FuelClasses(w)), eltype(FuelClasses(σ)), eltype(FuelClasses(h)), typeof(δ), typeof(Mx))
    Rothermel{T}(String(name), FuelClasses(w), FuelClasses(σ), FuelClasses(h), T(δ), T(Mx))
end

function Base.show(io::IO, r::Rothermel)
    print(io, "Rothermel{", eltype(r.w), "}(\"", r.name, "\")")
end

#-----------------------------------------------------------------------------# Constants
const _ρ_P = 32.0                       # Oven-dry particle density [lb/ft³]
const _S_T = 0.0555                     # Total mineral content [fraction]
const _S_E = 0.01                       # Effective mineral content [fraction]
const _TONS_ACRE_TO_LB_FT2 = 2000.0 / 43560.0
const _KMH_TO_FT_MIN = 3280.84 / 60.0  # km/h → ft/min
const _FT_TO_M = 0.3048                 # ft → m

#-----------------------------------------------------------------------------# rate_of_spread
"""
    rate_of_spread(fuel::Rothermel; moisture, wind, slope)

Compute the forward rate of fire spread using the Rothermel (1972) model.

# Arguments
- `fuel::Rothermel` - Fuel model
- `moisture::FuelClasses` - Moisture content per fuel class [fraction, 0–1].
  Use 0.0 for unused classes.
- `wind` - Midflame wind speed [km/h]
- `slope` - Terrain slope as rise/run [fraction]

# Returns
Forward rate of spread at the fire head [m/min].

# Example
```julia
M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
R = rate_of_spread(SHORT_GRASS, moisture=M, wind=8.0, slope=0.0)
```
"""
function rate_of_spread(fuel::Rothermel{T}; moisture::FuelClasses, wind, slope) where {T}
    (; w, σ, h, δ, Mx) = fuel
    M = moisture
    z = zero(T)
    δ > z || return z

    # --- Convert to internal units (US customary) ---
    w_i = map(x -> x * _TONS_ACRE_TO_LB_FT2, w)   # lb/ft²
    U   = wind * _KMH_TO_FT_MIN                     # ft/min

    # --- Surface area fractions ---
    a = map(*, σ, w_i)
    a = map(x -> x / _ρ_P, a)
    a_dead = a.d1 + a.d10 + a.d100
    a_live = a.herb + a.wood
    a_tot  = a_dead + a_live
    a_tot == z && return z

    f = FuelClasses(
        a_dead > z ? a.d1   / a_dead : z,
        a_dead > z ? a.d10  / a_dead : z,
        a_dead > z ? a.d100 / a_dead : z,
        a_live > z ? a.herb  / a_live : z,
        a_live > z ? a.wood  / a_live : z,
    )
    f_dead = a_dead / a_tot
    f_live = a_live / a_tot

    # --- Net fuel loading ---
    wn = map(x -> x * (1.0 - _S_T), w_i)
    wn_dead = f.d1*wn.d1 + f.d10*wn.d10 + f.d100*wn.d100
    wn_live = wn.herb + wn.wood

    # --- Characteristic fuel properties (area-weighted) ---
    mf_dead = f.d1*M.d1 + f.d10*M.d10 + f.d100*M.d100
    mf_live = f.herb*M.herb + f.wood*M.wood

    σ_dead = f.d1*σ.d1 + f.d10*σ.d10 + f.d100*σ.d100
    σ_live = f.herb*σ.herb + f.wood*σ.wood
    σ_tot  = f_dead * σ_dead + f_live * σ_live

    h_dead = f.d1*h.d1 + f.d10*h.d10 + f.d100*h.d100
    h_live = f.herb*h.herb + f.wood*h.wood

    # --- Packing ratio ---
    w_total = sum(x -> x / _ρ_P, w_i)
    β    = w_total / δ
    β_op = 3.348 * σ_tot^(-0.8189)
    rpr  = β / β_op   # relative packing ratio

    # --- Live fuel moisture of extinction (Albini 1976) ---
    if a_live > 0
        W_num = w_i.d1*exp(-138.0/σ.d1) + w_i.d10*exp(-138.0/σ.d10) + w_i.d100*exp(-138.0/σ.d100)
        W_den = w_i.herb*exp(-500.0/σ.herb) + w_i.wood*exp(-500.0/σ.wood)
        W = W_den > z ? W_num / W_den : z

        mfpd_num = w_i.d1*M.d1*exp(-138.0/σ.d1) + w_i.d10*M.d10*exp(-138.0/σ.d10) + w_i.d100*M.d100*exp(-138.0/σ.d100)
        mfpd = W_num > z ? mfpd_num / W_num : z

        mx_live = 2.9 * W * (1.0 - mfpd / Mx) - 0.226
        mx_live = max(mx_live, Mx)
    else
        mx_live = Mx
    end

    # --- Damping coefficients ---
    η_s = 0.174 * _S_E^(-0.19)   # mineral damping

    rm_dead  = Mx > z ? mf_dead / Mx : z
    η_M_dead = mf_dead >= Mx ? z : 1.0 - 2.59*rm_dead + 5.11*rm_dead^2 - 3.52*rm_dead^3

    if a_live > z && mx_live > z
        rm_live  = mf_live / mx_live
        η_M_live = mf_live >= mx_live ? z : 1.0 - 2.59*rm_live + 5.11*rm_live^2 - 3.52*rm_live^3
    else
        η_M_live = z
    end

    # --- Reaction intensity [BTU/ft²/min] ---
    A     = 133.0 * σ_tot^(-0.7913)
    Γ_max = σ_tot^1.5 / (495.0 + 0.0594 * σ_tot^1.5)
    Γ     = Γ_max * (rpr * exp(1.0 - rpr))^A
    I_R   = Γ * (wn_dead * h_dead * η_M_dead + wn_live * h_live * η_M_live) * η_s

    # --- Propagating flux ratio ---
    ξ = (192.0 + 0.2595 * σ_tot)^(-1) * exp((0.792 + 0.681 * sqrt(σ_tot)) * (β + 0.1))

    # --- Wind coefficient ---
    C   = 7.47 * exp(-0.133 * σ_tot^0.55)
    B   = 0.02526 * σ_tot^0.54
    E   = 0.715 * exp(-3.59e-4 * σ_tot)
    φ_w = U > z ? C * U^B * rpr^(-E) : z

    # --- Slope coefficient ---
    φ_s = 5.275 * β^(-0.3) * slope^2

    # --- Heat sink [BTU/ft³] ---
    ρ_b = sum(w_i) / δ   # bulk density [lb/ft³]

    eps = f_dead * (
            f.d1   * (250.0 + 1116.0*M.d1)   * exp(-138.0/σ.d1)   +
            f.d10  * (250.0 + 1116.0*M.d10)  * exp(-138.0/σ.d10)  +
            f.d100 * (250.0 + 1116.0*M.d100) * exp(-138.0/σ.d100)
          ) +
          f_live * (
            f.herb * (250.0 + 1116.0*M.herb) * exp(-138.0/σ.herb) +
            f.wood * (250.0 + 1116.0*M.wood) * exp(-138.0/σ.wood)
          )

    heat_sink = ρ_b * eps
    heat_sink ≤ z && return z

    # --- Rate of spread [ft/min → m/min] ---
    R = I_R * ξ * (1.0 + φ_w + φ_s) / heat_sink
    return R * _FT_TO_M
end

# ============================================================================ #
#               13 Standard NFFL Fuel Models — Anderson (1982)
# ============================================================================ #

const _H8 = FuelClasses(8000.0, 8000.0, 8000.0, 8000.0, 8000.0)
const _σ_STD = (109.0, 30.0, 1500.0, 1500.0)   # standard SAV for classes 2–5

# --- Grass (1–3) ---

"""NFFL 1: Short grass (1 ft)"""
const SHORT_GRASS = Rothermel("NFFL 1: Short grass (1 ft)",
    FuelClasses(0.74, 0.0,  0.0,  0.0,  0.0),  FuelClasses(3500.0, _σ_STD...), _H8, 1.0, 0.12)

"""NFFL 2: Timber grass and understory"""
const TIMBER_GRASS = Rothermel("NFFL 2: Timber grass/understory",
    FuelClasses(2.0,  1.0,  0.5,  0.5,  0.0),  FuelClasses(3000.0, _σ_STD...), _H8, 1.0, 0.15)

"""NFFL 3: Tall grass (2.5 ft)"""
const TALL_GRASS = Rothermel("NFFL 3: Tall grass (2.5 ft)",
    FuelClasses(3.01, 0.0,  0.0,  0.0,  0.0),  FuelClasses(1500.0, _σ_STD...), _H8, 2.5, 0.25)

# --- Chaparral/Shrub (4–7) ---

"""NFFL 4: Chaparral (6 ft)"""
const CHAPARRAL = Rothermel("NFFL 4: Chaparral (6 ft)",
    FuelClasses(5.01, 4.01, 2.0,  0.0,  5.01), FuelClasses(2000.0, _σ_STD...), _H8, 6.0, 0.20)

"""NFFL 5: Brush (2 ft)"""
const BRUSH = Rothermel("NFFL 5: Brush (2 ft)",
    FuelClasses(1.0,  0.5,  0.0,  0.0,  2.0),  FuelClasses(2000.0, _σ_STD...), _H8, 2.0, 0.20)

"""NFFL 6: Dormant brush, hardwood slash"""
const DORMANT_BRUSH = Rothermel("NFFL 6: Dormant brush/hardwood slash",
    FuelClasses(1.5,  2.5,  2.0,  0.0,  0.0),  FuelClasses(1750.0, _σ_STD...), _H8, 2.5, 0.25)

"""NFFL 7: Southern rough"""
const SOUTHERN_ROUGH = Rothermel("NFFL 7: Southern rough",
    FuelClasses(1.13, 1.87, 1.5,  0.0,  0.37), FuelClasses(1750.0, _σ_STD...), _H8, 2.5, 0.40)

# --- Timber Litter (8–10) ---

"""NFFL 8: Closed timber litter"""
const CLOSED_TIMBER_LITTER = Rothermel("NFFL 8: Closed timber litter",
    FuelClasses(1.5,  1.0,  2.5,  0.0,  0.0),  FuelClasses(2000.0, _σ_STD...), _H8, 0.2, 0.30)

"""NFFL 9: Hardwood litter"""
const HARDWOOD_LITTER = Rothermel("NFFL 9: Hardwood litter",
    FuelClasses(2.92, 0.41, 0.15, 0.0,  0.0),  FuelClasses(2500.0, _σ_STD...), _H8, 0.2, 0.25)

"""NFFL 10: Timber litter and understory"""
const TIMBER_UNDERSTORY = Rothermel("NFFL 10: Timber litter/understory",
    FuelClasses(3.01, 2.0,  5.01, 0.0,  2.0),  FuelClasses(2000.0, _σ_STD...), _H8, 1.0, 0.25)

# --- Logging Slash (11–13) ---

"""NFFL 11: Light logging slash"""
const LIGHT_SLASH = Rothermel("NFFL 11: Light logging slash",
    FuelClasses(1.5,  4.51, 5.51, 0.0,  0.0),  FuelClasses(1500.0, _σ_STD...), _H8, 1.0, 0.15)

"""NFFL 12: Medium logging slash"""
const MEDIUM_SLASH = Rothermel("NFFL 12: Medium logging slash",
    FuelClasses(4.01, 14.03, 16.53, 0.0, 0.0), FuelClasses(1500.0, _σ_STD...), _H8, 2.3, 0.20)

"""NFFL 13: Heavy logging slash"""
const HEAVY_SLASH = Rothermel("NFFL 13: Heavy logging slash",
    FuelClasses(7.01, 23.04, 28.05, 0.0, 0.0), FuelClasses(1500.0, _σ_STD...), _H8, 3.0, 0.25)

end # module
