module Rothermel

export FuelClasses, Rothermel, rate_of_spread, residence_time
export SHORT_GRASS, TIMBER_GRASS, TALL_GRASS, CHAPARRAL, BRUSH, DORMANT_BRUSH,
    SOUTHERN_ROUGH, CLOSED_TIMBER_LITTER, HARDWOOD_LITTER, TIMBER_UNDERSTORY,
    LIGHT_SLASH, MEDIUM_SLASH, HEAVY_SLASH
export GR1, GR2, GR3, GR4, GR5, GR6, GR7, GR8, GR9
export GS1, GS2, GS3, GS4
export SH1, SH2, SH3, SH4, SH5, SH6, SH7, SH8, SH9
export TU1, TU2, TU3, TU4, TU5
export TL1, TL2, TL3, TL4, TL5, TL6, TL7, TL8, TL9
export SB1, SB2, SB3, SB4

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
    o = one(T)
    δ > z || return z

    # --- Convert constants to T ---
    ρ_P = T(_ρ_P)
    S_T = T(_S_T)
    S_E = T(_S_E)
    tons_acre = T(_TONS_ACRE_TO_LB_FT2)
    kmh_ft   = T(_KMH_TO_FT_MIN)
    ft_m     = T(_FT_TO_M)

    # --- Convert to internal units (US customary) ---
    w_i = map(x -> x * tons_acre, w)   # lb/ft²
    U   = wind * kmh_ft                  # ft/min

    # --- Surface area fractions ---
    a = map(*, σ, w_i)
    a = map(x -> x / ρ_P, a)
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
    wn = map(x -> x * (o - S_T), w_i)
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
    w_total = sum(x -> x / ρ_P, w_i)
    β    = w_total / δ
    β_op = T(3.348) * σ_tot^T(-0.8189)
    rpr  = β / β_op   # relative packing ratio

    # --- Live fuel moisture of extinction (Albini 1976) ---
    if a_live > 0
        W_num = w_i.d1*exp(T(-138)/σ.d1) + w_i.d10*exp(T(-138)/σ.d10) + w_i.d100*exp(T(-138)/σ.d100)
        W_den = w_i.herb*exp(T(-500)/σ.herb) + w_i.wood*exp(T(-500)/σ.wood)
        W = W_den > z ? W_num / W_den : z

        mfpd_num = w_i.d1*M.d1*exp(T(-138)/σ.d1) + w_i.d10*M.d10*exp(T(-138)/σ.d10) + w_i.d100*M.d100*exp(T(-138)/σ.d100)
        mfpd = W_num > z ? mfpd_num / W_num : z

        mx_live = T(2.9) * W * (o - mfpd / Mx) - T(0.226)
        mx_live = max(mx_live, Mx)
    else
        mx_live = Mx
    end

    # --- Damping coefficients ---
    η_s = T(0.174) * S_E^T(-0.19)   # mineral damping

    rm_dead  = Mx > z ? mf_dead / Mx : z
    η_M_dead = mf_dead >= Mx ? z : o - T(2.59)*rm_dead + T(5.11)*rm_dead^2 - T(3.52)*rm_dead^3

    if a_live > z && mx_live > z
        rm_live  = mf_live / mx_live
        η_M_live = mf_live >= mx_live ? z : o - T(2.59)*rm_live + T(5.11)*rm_live^2 - T(3.52)*rm_live^3
    else
        η_M_live = z
    end

    # --- Reaction intensity [BTU/ft²/min] ---
    A     = T(133) * σ_tot^T(-0.7913)
    Γ_max = σ_tot^T(1.5) / (T(495) + T(0.0594) * σ_tot^T(1.5))
    Γ     = Γ_max * (rpr * exp(o - rpr))^A
    I_R   = Γ * (wn_dead * h_dead * η_M_dead + wn_live * h_live * η_M_live) * η_s

    # --- Propagating flux ratio ---
    ξ = (T(192) + T(0.2595) * σ_tot)^(-o) * exp((T(0.792) + T(0.681) * sqrt(σ_tot)) * (β + T(0.1)))

    # --- Wind coefficient ---
    C   = T(7.47) * exp(T(-0.133) * σ_tot^T(0.55))
    B   = T(0.02526) * σ_tot^T(0.54)
    E   = T(0.715) * exp(T(-3.59e-4) * σ_tot)
    φ_w = U > z ? C * U^B * rpr^(-E) : z

    # --- Slope coefficient ---
    φ_s = T(5.275) * β^T(-0.3) * slope^2

    # --- Heat sink [BTU/ft³] ---
    ρ_b = sum(w_i) / δ   # bulk density [lb/ft³]

    eps = f_dead * (
            f.d1   * (T(250) + T(1116)*M.d1)   * exp(T(-138)/σ.d1)   +
            f.d10  * (T(250) + T(1116)*M.d10)  * exp(T(-138)/σ.d10)  +
            f.d100 * (T(250) + T(1116)*M.d100) * exp(T(-138)/σ.d100)
          ) +
          f_live * (
            f.herb * (T(250) + T(1116)*M.herb) * exp(T(-138)/σ.herb) +
            f.wood * (T(250) + T(1116)*M.wood) * exp(T(-138)/σ.wood)
          )

    heat_sink = ρ_b * eps
    heat_sink ≤ z && return z

    # --- Rate of spread [ft/min → m/min] ---
    R = I_R * ξ * (o + φ_w + φ_s) / heat_sink
    return R * ft_m
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

# ============================================================================ #
#      40 Scott & Burgan (2005) Fire Behavior Fuel Models — GTR-153/371
# ============================================================================ #
#
# Source: Scott, J.H. & Burgan, R.E. (2005). Standard Fire Behavior Fuel Models:
#   A Comprehensive Set for Use with Rothermel's Surface Fire Spread Model.
#   Gen. Tech. Rep. RMRS-GTR-153, USDA Forest Service.
#
# Parameters from Andrews (2018) Table 8, GTR-371.
# Constants for all models: 10-h SAV = 109 ft²/ft³, 100-h SAV = 30 ft²/ft³,
#   heat content = 8000 BTU/lb, particle density = 32 lb/ft³,
#   total mineral content = 0.0555, effective mineral content = 0.010.
# Fuel loads are tons/acre (pre-transfer values for dynamic models).
# Dynamic models (D) transfer live herbaceous fuel to dead based on curing;
#   that transfer is handled outside the Rothermel calculation.

# --- GR: Grass (101–109) ---

"""SB40 GR1 (101): Short, sparse, dry climate grass [Dynamic]"""
const GR1 = Rothermel("SB40 GR1: Short, sparse, dry climate grass",
    FuelClasses(0.10, 0.0, 0.0, 0.30, 0.0), FuelClasses(2200.0, 109.0, 30.0, 2000.0, 1500.0), _H8, 0.4, 0.15)

"""SB40 GR2 (102): Low load, dry climate grass [Dynamic]"""
const GR2 = Rothermel("SB40 GR2: Low load, dry climate grass",
    FuelClasses(0.10, 0.0, 0.0, 1.00, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 1.0, 0.15)

"""SB40 GR3 (103): Low load, very coarse, humid climate grass [Dynamic]"""
const GR3 = Rothermel("SB40 GR3: Low load, very coarse, humid climate grass",
    FuelClasses(0.10, 0.40, 0.0, 1.50, 0.0), FuelClasses(1500.0, 109.0, 30.0, 1300.0, 1500.0), _H8, 2.0, 0.30)

"""SB40 GR4 (104): Moderate load, dry climate grass [Dynamic]"""
const GR4 = Rothermel("SB40 GR4: Moderate load, dry climate grass",
    FuelClasses(0.25, 0.0, 0.0, 1.90, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 2.0, 0.15)

"""SB40 GR5 (105): Low load, humid climate grass [Dynamic]"""
const GR5 = Rothermel("SB40 GR5: Low load, humid climate grass",
    FuelClasses(0.40, 0.0, 0.0, 2.50, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1500.0), _H8, 1.5, 0.40)

"""SB40 GR6 (106): Moderate load, humid climate grass [Dynamic]"""
const GR6 = Rothermel("SB40 GR6: Moderate load, humid climate grass",
    FuelClasses(0.10, 0.0, 0.0, 3.40, 0.0), FuelClasses(2200.0, 109.0, 30.0, 2000.0, 1500.0), _H8, 1.5, 0.40)

"""SB40 GR7 (107): High load, dry climate grass [Dynamic]"""
const GR7 = Rothermel("SB40 GR7: High load, dry climate grass",
    FuelClasses(1.00, 0.0, 0.0, 5.40, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 3.0, 0.15)

"""SB40 GR8 (108): High load, very coarse, humid climate grass [Dynamic]"""
const GR8 = Rothermel("SB40 GR8: High load, very coarse, humid climate grass",
    FuelClasses(0.50, 1.00, 0.0, 7.30, 0.0), FuelClasses(1500.0, 109.0, 30.0, 1300.0, 1500.0), _H8, 4.0, 0.30)

"""SB40 GR9 (109): Very high load, humid climate grass [Dynamic]"""
const GR9 = Rothermel("SB40 GR9: Very high load, humid climate grass",
    FuelClasses(1.00, 1.00, 0.0, 9.00, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1500.0), _H8, 5.0, 0.40)

# --- GS: Grass-Shrub (121–124) ---

"""SB40 GS1 (121): Low load, dry climate grass-shrub [Dynamic]"""
const GS1 = Rothermel("SB40 GS1: Low load, dry climate grass-shrub",
    FuelClasses(0.20, 0.0, 0.0, 0.50, 0.65), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1800.0), _H8, 0.9, 0.15)

"""SB40 GS2 (122): Moderate load, dry climate grass-shrub [Dynamic]"""
const GS2 = Rothermel("SB40 GS2: Moderate load, dry climate grass-shrub",
    FuelClasses(0.50, 0.50, 0.0, 0.60, 1.00), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1800.0), _H8, 1.5, 0.15)

"""SB40 GS3 (123): Moderate load, humid climate grass-shrub [Dynamic]"""
const GS3 = Rothermel("SB40 GS3: Moderate load, humid climate grass-shrub",
    FuelClasses(0.30, 0.25, 0.0, 1.45, 1.25), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1600.0), _H8, 1.8, 0.40)

"""SB40 GS4 (124): High load, humid climate grass-shrub [Dynamic]"""
const GS4 = Rothermel("SB40 GS4: High load, humid climate grass-shrub",
    FuelClasses(1.90, 0.30, 0.10, 3.40, 7.10), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1600.0), _H8, 2.1, 0.40)

# --- SH: Shrub (141–149) ---

"""SB40 SH1 (141): Low load, dry climate shrub [Dynamic]"""
const SH1 = Rothermel("SB40 SH1: Low load, dry climate shrub",
    FuelClasses(0.25, 0.25, 0.0, 0.15, 1.30), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1600.0), _H8, 1.0, 0.15)

"""SB40 SH2 (142): Moderate load, dry climate shrub [Static]"""
const SH2 = Rothermel("SB40 SH2: Moderate load, dry climate shrub",
    FuelClasses(1.35, 2.40, 0.75, 0.0, 3.85), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 1.0, 0.15)

"""SB40 SH3 (143): Moderate load, humid climate shrub [Static]"""
const SH3 = Rothermel("SB40 SH3: Moderate load, humid climate shrub",
    FuelClasses(0.45, 3.00, 0.0, 0.0, 6.20), FuelClasses(1600.0, 109.0, 30.0, 1500.0, 1400.0), _H8, 2.4, 0.40)

"""SB40 SH4 (144): Low load, humid climate timber-shrub [Static]"""
const SH4 = Rothermel("SB40 SH4: Low load, humid climate timber-shrub",
    FuelClasses(0.85, 1.15, 0.20, 0.0, 2.55), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 3.0, 0.30)

"""SB40 SH5 (145): High load, dry climate shrub [Static]"""
const SH5 = Rothermel("SB40 SH5: High load, dry climate shrub",
    FuelClasses(3.60, 2.10, 0.0, 0.0, 2.90), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 6.0, 0.15)

"""SB40 SH6 (146): Low load, humid climate shrub [Static]"""
const SH6 = Rothermel("SB40 SH6: Low load, humid climate shrub",
    FuelClasses(2.90, 1.45, 0.0, 0.0, 1.40), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 2.0, 0.30)

"""SB40 SH7 (147): Very high load, dry climate shrub [Static]"""
const SH7 = Rothermel("SB40 SH7: Very high load, dry climate shrub",
    FuelClasses(3.50, 5.30, 2.20, 0.0, 3.40), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 6.0, 0.15)

"""SB40 SH8 (148): High load, humid climate shrub [Static]"""
const SH8 = Rothermel("SB40 SH8: High load, humid climate shrub",
    FuelClasses(2.05, 3.40, 0.85, 0.0, 4.35), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 3.0, 0.40)

"""SB40 SH9 (149): Very high load, humid climate shrub [Dynamic]"""
const SH9 = Rothermel("SB40 SH9: Very high load, humid climate shrub",
    FuelClasses(4.50, 2.45, 0.0, 1.55, 7.00), FuelClasses(750.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 4.4, 0.40)

# --- TU: Timber-Understory (161–165) ---

"""SB40 TU1 (161): Light load, dry climate timber-grass-shrub [Dynamic]"""
const TU1 = Rothermel("SB40 TU1: Light load, dry climate timber-grass-shrub",
    FuelClasses(0.20, 0.90, 1.50, 0.20, 0.90), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1600.0), _H8, 0.6, 0.20)

"""SB40 TU2 (162): Moderate load, humid climate timber-shrub [Static]"""
const TU2 = Rothermel("SB40 TU2: Moderate load, humid climate timber-shrub",
    FuelClasses(0.95, 1.80, 1.25, 0.0, 0.20), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 1.0, 0.30)

"""SB40 TU3 (163): Moderate load, humid climate timber-grass-shrub [Dynamic]"""
const TU3 = Rothermel("SB40 TU3: Moderate load, humid climate timber-grass-shrub",
    FuelClasses(1.10, 0.15, 0.25, 0.65, 1.10), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1400.0), _H8, 1.3, 0.30)

"""SB40 TU4 (164): Dwarf conifer understory [Static]"""
const TU4 = Rothermel("SB40 TU4: Dwarf conifer understory",
    FuelClasses(4.50, 0.0, 0.0, 0.0, 2.00), FuelClasses(2300.0, 109.0, 30.0, 1500.0, 2000.0), _H8, 0.5, 0.12)

"""SB40 TU5 (165): Very high load, dry climate timber-shrub [Static]"""
const TU5 = Rothermel("SB40 TU5: Very high load, dry climate timber-shrub",
    FuelClasses(4.00, 4.00, 3.00, 0.0, 3.00), FuelClasses(1500.0, 109.0, 30.0, 1500.0, 750.0), _H8, 1.0, 0.25)

# --- TL: Timber Litter (181–189) ---

"""SB40 TL1 (181): Low load, compact conifer litter [Static]"""
const TL1 = Rothermel("SB40 TL1: Low load, compact conifer litter",
    FuelClasses(1.00, 2.20, 3.60, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.2, 0.30)

"""SB40 TL2 (182): Low broadleaf litter [Static]"""
const TL2 = Rothermel("SB40 TL2: Low broadleaf litter",
    FuelClasses(1.40, 2.30, 2.20, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.2, 0.25)

"""SB40 TL3 (183): Moderate load conifer litter [Static]"""
const TL3 = Rothermel("SB40 TL3: Moderate load conifer litter",
    FuelClasses(0.50, 2.20, 2.80, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.20)

"""SB40 TL4 (184): Small downed logs [Static]"""
const TL4 = Rothermel("SB40 TL4: Small downed logs",
    FuelClasses(0.50, 1.50, 4.20, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.4, 0.25)

"""SB40 TL5 (185): High load conifer litter [Static]"""
const TL5 = Rothermel("SB40 TL5: High load conifer litter",
    FuelClasses(1.15, 2.50, 4.40, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.6, 0.25)

"""SB40 TL6 (186): Moderate load broadleaf litter [Static]"""
const TL6 = Rothermel("SB40 TL6: Moderate load broadleaf litter",
    FuelClasses(2.40, 1.20, 1.20, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.25)

"""SB40 TL7 (187): Large downed logs [Static]"""
const TL7 = Rothermel("SB40 TL7: Large downed logs",
    FuelClasses(0.30, 1.40, 8.10, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.4, 0.25)

"""SB40 TL8 (188): Long-needle litter [Static]"""
const TL8 = Rothermel("SB40 TL8: Long-needle litter",
    FuelClasses(5.80, 1.40, 1.10, 0.0, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.35)

"""SB40 TL9 (189): Very high load broadleaf litter [Static]"""
const TL9 = Rothermel("SB40 TL9: Very high load broadleaf litter",
    FuelClasses(6.65, 3.30, 4.15, 0.0, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.6, 0.35)

# --- SB: Slash-Blowdown (201–204) ---

"""SB40 SB1 (201): Low activity fuel [Static]"""
const SB1 = Rothermel("SB40 SB1: Low activity fuel",
    FuelClasses(1.50, 3.00, 11.00, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.0, 0.25)

"""SB40 SB2 (202): Moderate activity fuel or low load blowdown [Static]"""
const SB2 = Rothermel("SB40 SB2: Moderate activity fuel or low load blowdown",
    FuelClasses(4.50, 4.25, 4.00, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.0, 0.25)

"""SB40 SB3 (203): High activity fuel or moderate load blowdown [Static]"""
const SB3 = Rothermel("SB40 SB3: High activity fuel or moderate load blowdown",
    FuelClasses(5.50, 2.75, 3.00, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.2, 0.25)

"""SB40 SB4 (204): High load blowdown [Static]"""
const SB4 = Rothermel("SB40 SB4: High load blowdown",
    FuelClasses(5.25, 3.50, 5.25, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 2.7, 0.25)

#-----------------------------------------------------------------------------# residence_time
"""
    residence_time(fuel::Rothermel)

Flame residence time [min] from Anderson (1969): `t_r = 384 / σ_char` seconds,
converted to minutes.

Uses the same characteristic SAV computation as [`rate_of_spread`](@ref).

### Examples
```julia
residence_time(SHORT_GRASS)  # ≈ 0.00183 min
```
"""
function residence_time(fuel::Rothermel{T}) where {T}
    (; w, σ) = fuel
    z = zero(T)
    tons_acre = T(_TONS_ACRE_TO_LB_FT2)
    ρ_P = T(_ρ_P)

    w_i = map(x -> x * tons_acre, w)
    a = map(x -> x / ρ_P, map(*, σ, w_i))
    a_dead = a.d1 + a.d10 + a.d100
    a_live = a.herb + a.wood
    a_tot = a_dead + a_live
    a_tot > z || return T(Inf)

    f = FuelClasses(
        a_dead > z ? a.d1 / a_dead : z,
        a_dead > z ? a.d10 / a_dead : z,
        a_dead > z ? a.d100 / a_dead : z,
        a_live > z ? a.herb / a_live : z,
        a_live > z ? a.wood / a_live : z,
    )
    σ_dead = f.d1*σ.d1 + f.d10*σ.d10 + f.d100*σ.d100
    σ_live = f.herb*σ.herb + f.wood*σ.wood
    σ_tot = (a_dead / a_tot) * σ_dead + (a_live / a_tot) * σ_live
    σ_tot > z || return T(Inf)

    return T(384) / (σ_tot * T(60))
end

end # module
