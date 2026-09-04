module RateOfSpread

export AbstractRateOfSpread, rate_of_spread
export FuelClasses

# Rothermel
export RothermelROS, RothermelFuel, residence_time
export SHORT_GRASS, TIMBER_GRASS, TALL_GRASS, CHAPARRAL, BRUSH, DORMANT_BRUSH,
    SOUTHERN_ROUGH, CLOSED_TIMBER_LITTER, HARDWOOD_LITTER, TIMBER_UNDERSTORY,
    LIGHT_SLASH, MEDIUM_SLASH, HEAVY_SLASH
export NFFL_MODELS, nffl_model
export GR1, GR2, GR3, GR4, GR5, GR6, GR7, GR8, GR9
export GS1, GS2, GS3, GS4
export SH1, SH2, SH3, SH4, SH5, SH6, SH7, SH8, SH9
export TU1, TU2, TU3, TU4, TU5
export TL1, TL2, TL3, TL4, TL5, TL6, TL7, TL8, TL9
export SB1, SB2, SB3, SB4

# McArthur
export McArthurGrasslandROS, McArthurForestROS

# Canadian FBP
export CanadianFBP_ROS, FBPFuelType
export FBP_C1, FBP_C2, FBP_C3, FBP_C4, FBP_C5, FBP_C6, FBP_C7
export FBP_D1, FBP_M3, FBP_M4
export FBP_S1, FBP_S2, FBP_S3
export FBP_O1a, FBP_O1b

#-----------------------------------------------------------------------------# AbstractRateOfSpread
"""
    AbstractRateOfSpread

Supertype for rate-of-spread models.

All subtypes must implement:

    rate_of_spread(model::AbstractRateOfSpread; kwargs...) -> T  [m/min]

Each model defines its own keyword arguments (moisture, wind, slope, etc.).
The return type matches the numeric type of the model/inputs.
"""
abstract type AbstractRateOfSpread end

"""
    rate_of_spread(model::AbstractRateOfSpread; kwargs...) -> T

Compute the forward rate of fire spread [m/min].

Keyword arguments vary by model type.  See subtypes for details.
"""
function rate_of_spread end

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

# ============================================================================ #
#                          Rothermel (1972)
# ============================================================================ #

#-----------------------------------------------------------------------------# RothermelFuel
"""
    RothermelFuel{T}(; name, w, σ, h, δ, Mx)

Fuel model for the Rothermel (1972) surface fire spread model.

# Fields (US customary units, matching original publications)
- `name::String` - Description
- `w::FuelClasses{T}` - Fuel loading [tons/acre]
- `σ::FuelClasses{T}` - Surface-area-to-volume ratio [1/ft]
- `h::FuelClasses{T}` - Heat content [BTU/lb]
- `δ::T` - Fuel bed depth [ft]
- `Mx::T` - Dead fuel moisture of extinction [fraction]
"""
Base.@kwdef struct RothermelFuel{T}
    name::String = "Custom"
    w::FuelClasses{T}
    σ::FuelClasses{T}
    h::FuelClasses{T}
    δ::T
    Mx::T
end

function RothermelFuel(name::AbstractString, w, σ, h, δ, Mx)
    T = promote_type(eltype(FuelClasses(w)), eltype(FuelClasses(σ)), eltype(FuelClasses(h)), typeof(δ), typeof(Mx))
    RothermelFuel{T}(String(name), FuelClasses(w), FuelClasses(σ), FuelClasses(h), T(δ), T(Mx))
end

function Base.show(io::IO, r::RothermelFuel)
    print(io, "RothermelFuel{", eltype(r.w), "}(\"", r.name, "\")")
end

#-----------------------------------------------------------------------------# RothermelROS
"""
    RothermelROS(fuel::RothermelFuel)

Rothermel (1972) rate of spread model.

# Rate of Spread
    rate_of_spread(model::RothermelROS; moisture::FuelClasses, wind, slope) -> Float64  [m/min]

- `moisture` — Moisture content per fuel class [fraction, 0–1]
- `wind` — Midflame wind speed [km/h]
- `slope` — Terrain slope as rise/run [fraction]

# References
- Rothermel, R.C. (1972). A Mathematical Model for Predicting Fire Spread in
  Wildland Fuels. Res. Paper INT-115, USDA Forest Service.
- Andrews, P.L. (2018). The Rothermel Surface Fire Spread Model and Associated
  Developments. Gen. Tech. Rep. RMRS-GTR-371, USDA Forest Service.

### Examples
```julia
M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
model = RothermelROS(SHORT_GRASS)
rate_of_spread(model; moisture=M, wind=8.0, slope=0.0)
```
"""
struct RothermelROS{T} <: AbstractRateOfSpread
    fuel::RothermelFuel{T}
end

#-----------------------------------------------------------------------------# Rothermel Constants
const _ρ_P = 32.0                       # Oven-dry particle density [lb/ft³]
const _S_T = 0.0555                     # Total mineral content [fraction]
const _S_E = 0.01                       # Effective mineral content [fraction]
const _TONS_ACRE_TO_LB_FT2 = 2000.0 / 43560.0
const _KMH_TO_FT_MIN = 3280.84 / 60.0  # km/h → ft/min
const _FT_TO_M = 0.3048                 # ft → m

function rate_of_spread(model::RothermelROS{T}; moisture::FuelClasses, wind, slope) where {T}
    (; w, σ, h, δ, Mx) = model.fuel
    M = moisture
    z = zero(T)
    o = one(T)
    δ > z || return z

    ρ_P = T(_ρ_P)
    S_T = T(_S_T)
    S_E = T(_S_E)
    tons_acre = T(_TONS_ACRE_TO_LB_FT2)
    kmh_ft   = T(_KMH_TO_FT_MIN)
    ft_m     = T(_FT_TO_M)

    w_i = map(x -> x * tons_acre, w)
    U   = wind * kmh_ft

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

    wn = map(x -> x * (o - S_T), w_i)
    wn_dead = f.d1*wn.d1 + f.d10*wn.d10 + f.d100*wn.d100
    wn_live = wn.herb + wn.wood

    mf_dead = f.d1*M.d1 + f.d10*M.d10 + f.d100*M.d100
    mf_live = f.herb*M.herb + f.wood*M.wood

    σ_dead = f.d1*σ.d1 + f.d10*σ.d10 + f.d100*σ.d100
    σ_live = f.herb*σ.herb + f.wood*σ.wood
    σ_tot  = f_dead * σ_dead + f_live * σ_live

    h_dead = f.d1*h.d1 + f.d10*h.d10 + f.d100*h.d100
    h_live = f.herb*h.herb + f.wood*h.wood

    w_total = sum(x -> x / ρ_P, w_i)
    β    = w_total / δ
    β_op = T(3.348) * σ_tot^T(-0.8189)
    rpr  = β / β_op

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

    η_s = T(0.174) * S_E^T(-0.19)
    rm_dead  = Mx > z ? mf_dead / Mx : z
    η_M_dead = mf_dead >= Mx ? z : o - T(2.59)*rm_dead + T(5.11)*rm_dead^2 - T(3.52)*rm_dead^3

    if a_live > z && mx_live > z
        rm_live  = mf_live / mx_live
        η_M_live = mf_live >= mx_live ? z : o - T(2.59)*rm_live + T(5.11)*rm_live^2 - T(3.52)*rm_live^3
    else
        η_M_live = z
    end

    A     = T(133) * σ_tot^T(-0.7913)
    Γ_max = σ_tot^T(1.5) / (T(495) + T(0.0594) * σ_tot^T(1.5))
    Γ     = Γ_max * (rpr * exp(o - rpr))^A
    I_R   = Γ * (wn_dead * h_dead * η_M_dead + wn_live * h_live * η_M_live) * η_s

    ξ = (T(192) + T(0.2595) * σ_tot)^(-o) * exp((T(0.792) + T(0.681) * sqrt(σ_tot)) * (β + T(0.1)))

    C   = T(7.47) * exp(T(-0.133) * σ_tot^T(0.55))
    B   = T(0.02526) * σ_tot^T(0.54)
    E   = T(0.715) * exp(T(-3.59e-4) * σ_tot)
    φ_w = U > z ? C * U^B * rpr^(-E) : z

    φ_s = T(5.275) * β^T(-0.3) * slope^2

    ρ_b = sum(w_i) / δ
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

    R = I_R * ξ * (o + φ_w + φ_s) / heat_sink
    return R * ft_m
end

#-----------------------------------------------------------------------------# residence_time
"""
    residence_time(fuel::RothermelFuel)

Flame residence time [min] from Anderson (1969): `t_r = 384 / σ_char` seconds,
converted to minutes.

### Examples
```julia
residence_time(SHORT_GRASS)  # ≈ 0.00183 min
```
"""
function residence_time(fuel::RothermelFuel{T}) where {T}
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

# ============================================================================ #
#              NFFL Fuel Models — Anderson (1982)
# ============================================================================ #

const _H8 = FuelClasses(8000.0, 8000.0, 8000.0, 8000.0, 8000.0)
const _σ_STD = (109.0, 30.0, 1500.0, 1500.0)

"""NFFL 1: Short grass (1 ft)"""
const SHORT_GRASS = RothermelFuel("NFFL 1: Short grass (1 ft)",
    FuelClasses(0.74, 0.0,  0.0,  0.0,  0.0),  FuelClasses(3500.0, _σ_STD...), _H8, 1.0, 0.12)

"""NFFL 2: Timber grass and understory"""
const TIMBER_GRASS = RothermelFuel("NFFL 2: Timber grass/understory",
    FuelClasses(2.0,  1.0,  0.5,  0.5,  0.0),  FuelClasses(3000.0, _σ_STD...), _H8, 1.0, 0.15)

"""NFFL 3: Tall grass (2.5 ft)"""
const TALL_GRASS = RothermelFuel("NFFL 3: Tall grass (2.5 ft)",
    FuelClasses(3.01, 0.0,  0.0,  0.0,  0.0),  FuelClasses(1500.0, _σ_STD...), _H8, 2.5, 0.25)

"""NFFL 4: Chaparral (6 ft)"""
const CHAPARRAL = RothermelFuel("NFFL 4: Chaparral (6 ft)",
    FuelClasses(5.01, 4.01, 2.0,  0.0,  5.01), FuelClasses(2000.0, _σ_STD...), _H8, 6.0, 0.20)

"""NFFL 5: Brush (2 ft)"""
const BRUSH = RothermelFuel("NFFL 5: Brush (2 ft)",
    FuelClasses(1.0,  0.5,  0.0,  0.0,  2.0),  FuelClasses(2000.0, _σ_STD...), _H8, 2.0, 0.20)

"""NFFL 6: Dormant brush, hardwood slash"""
const DORMANT_BRUSH = RothermelFuel("NFFL 6: Dormant brush/hardwood slash",
    FuelClasses(1.5,  2.5,  2.0,  0.0,  0.0),  FuelClasses(1750.0, _σ_STD...), _H8, 2.5, 0.25)

"""NFFL 7: Southern rough"""
const SOUTHERN_ROUGH = RothermelFuel("NFFL 7: Southern rough",
    FuelClasses(1.13, 1.87, 1.5,  0.0,  0.37), FuelClasses(1750.0, _σ_STD...), _H8, 2.5, 0.40)

"""NFFL 8: Closed timber litter"""
const CLOSED_TIMBER_LITTER = RothermelFuel("NFFL 8: Closed timber litter",
    FuelClasses(1.5,  1.0,  2.5,  0.0,  0.0),  FuelClasses(2000.0, _σ_STD...), _H8, 0.2, 0.30)

"""NFFL 9: Hardwood litter"""
const HARDWOOD_LITTER = RothermelFuel("NFFL 9: Hardwood litter",
    FuelClasses(2.92, 0.41, 0.15, 0.0,  0.0),  FuelClasses(2500.0, _σ_STD...), _H8, 0.2, 0.25)

"""NFFL 10: Timber litter and understory"""
const TIMBER_UNDERSTORY = RothermelFuel("NFFL 10: Timber litter/understory",
    FuelClasses(3.01, 2.0,  5.01, 0.0,  2.0),  FuelClasses(2000.0, _σ_STD...), _H8, 1.0, 0.25)

"""NFFL 11: Light logging slash"""
const LIGHT_SLASH = RothermelFuel("NFFL 11: Light logging slash",
    FuelClasses(1.5,  4.51, 5.51, 0.0,  0.0),  FuelClasses(1500.0, _σ_STD...), _H8, 1.0, 0.15)

"""NFFL 12: Medium logging slash"""
const MEDIUM_SLASH = RothermelFuel("NFFL 12: Medium logging slash",
    FuelClasses(4.01, 14.03, 16.53, 0.0, 0.0), FuelClasses(1500.0, _σ_STD...), _H8, 2.3, 0.20)

"""NFFL 13: Heavy logging slash"""
const HEAVY_SLASH = RothermelFuel("NFFL 13: Heavy logging slash",
    FuelClasses(7.01, 23.04, 28.05, 0.0, 0.0), FuelClasses(1500.0, _σ_STD...), _H8, 3.0, 0.25)

const NFFL_MODELS = Dict{Int, RothermelFuel{Float64}}(
    1  => SHORT_GRASS, 2  => TIMBER_GRASS, 3  => TALL_GRASS,
    4  => CHAPARRAL,   5  => BRUSH,        6  => DORMANT_BRUSH,
    7  => SOUTHERN_ROUGH,
    8  => CLOSED_TIMBER_LITTER, 9 => HARDWOOD_LITTER, 10 => TIMBER_UNDERSTORY,
    11 => LIGHT_SLASH, 12 => MEDIUM_SLASH, 13 => HEAVY_SLASH,
)

"""
    nffl_model(code::Integer) -> Union{RothermelFuel, Nothing}

Look up an NFFL fuel model by code (1–13). Returns `nothing` for unknown codes.
"""
nffl_model(code::Integer) = get(NFFL_MODELS, Int(code), nothing)

# ============================================================================ #
#        Scott & Burgan (2005) Fuel Models — GTR-153/371
# ============================================================================ #

"""SB40 GR1 (101): Short, sparse, dry climate grass"""
const GR1 = RothermelFuel("SB40 GR1: Short, sparse, dry climate grass",
    FuelClasses(0.10, 0.0, 0.0, 0.30, 0.0), FuelClasses(2200.0, 109.0, 30.0, 2000.0, 1500.0), _H8, 0.4, 0.15)
"""SB40 GR2 (102): Low load, dry climate grass"""
const GR2 = RothermelFuel("SB40 GR2: Low load, dry climate grass",
    FuelClasses(0.10, 0.0, 0.0, 1.00, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 1.0, 0.15)
"""SB40 GR3 (103): Low load, very coarse, humid climate grass"""
const GR3 = RothermelFuel("SB40 GR3: Low load, very coarse, humid climate grass",
    FuelClasses(0.10, 0.40, 0.0, 1.50, 0.0), FuelClasses(1500.0, 109.0, 30.0, 1300.0, 1500.0), _H8, 2.0, 0.30)
"""SB40 GR4 (104): Moderate load, dry climate grass"""
const GR4 = RothermelFuel("SB40 GR4: Moderate load, dry climate grass",
    FuelClasses(0.25, 0.0, 0.0, 1.90, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 2.0, 0.15)
"""SB40 GR5 (105): Low load, humid climate grass"""
const GR5 = RothermelFuel("SB40 GR5: Low load, humid climate grass",
    FuelClasses(0.40, 0.0, 0.0, 2.50, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1500.0), _H8, 1.5, 0.40)
"""SB40 GR6 (106): Moderate load, humid climate grass"""
const GR6 = RothermelFuel("SB40 GR6: Moderate load, humid climate grass",
    FuelClasses(0.10, 0.0, 0.0, 3.40, 0.0), FuelClasses(2200.0, 109.0, 30.0, 2000.0, 1500.0), _H8, 1.5, 0.40)
"""SB40 GR7 (107): High load, dry climate grass"""
const GR7 = RothermelFuel("SB40 GR7: High load, dry climate grass",
    FuelClasses(1.00, 0.0, 0.0, 5.40, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 3.0, 0.15)
"""SB40 GR8 (108): High load, very coarse, humid climate grass"""
const GR8 = RothermelFuel("SB40 GR8: High load, very coarse, humid climate grass",
    FuelClasses(0.50, 1.00, 0.0, 7.30, 0.0), FuelClasses(1500.0, 109.0, 30.0, 1300.0, 1500.0), _H8, 4.0, 0.30)
"""SB40 GR9 (109): Very high load, humid climate grass"""
const GR9 = RothermelFuel("SB40 GR9: Very high load, humid climate grass",
    FuelClasses(1.00, 1.00, 0.0, 9.00, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1500.0), _H8, 5.0, 0.40)

"""SB40 GS1 (121): Low load, dry climate grass-shrub"""
const GS1 = RothermelFuel("SB40 GS1: Low load, dry climate grass-shrub",
    FuelClasses(0.20, 0.0, 0.0, 0.50, 0.65), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1800.0), _H8, 0.9, 0.15)
"""SB40 GS2 (122): Moderate load, dry climate grass-shrub"""
const GS2 = RothermelFuel("SB40 GS2: Moderate load, dry climate grass-shrub",
    FuelClasses(0.50, 0.50, 0.0, 0.60, 1.00), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1800.0), _H8, 1.5, 0.15)
"""SB40 GS3 (123): Moderate load, humid climate grass-shrub"""
const GS3 = RothermelFuel("SB40 GS3: Moderate load, humid climate grass-shrub",
    FuelClasses(0.30, 0.25, 0.0, 1.45, 1.25), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1600.0), _H8, 1.8, 0.40)
"""SB40 GS4 (124): High load, humid climate grass-shrub"""
const GS4 = RothermelFuel("SB40 GS4: High load, humid climate grass-shrub",
    FuelClasses(1.90, 0.30, 0.10, 3.40, 7.10), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1600.0), _H8, 2.1, 0.40)

"""SB40 SH1 (141): Low load, dry climate shrub"""
const SH1 = RothermelFuel("SB40 SH1: Low load, dry climate shrub",
    FuelClasses(0.25, 0.25, 0.0, 0.15, 1.30), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1600.0), _H8, 1.0, 0.15)
"""SB40 SH2 (142): Moderate load, dry climate shrub"""
const SH2 = RothermelFuel("SB40 SH2: Moderate load, dry climate shrub",
    FuelClasses(1.35, 2.40, 0.75, 0.0, 3.85), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 1.0, 0.15)
"""SB40 SH3 (143): Moderate load, humid climate shrub"""
const SH3 = RothermelFuel("SB40 SH3: Moderate load, humid climate shrub",
    FuelClasses(0.45, 3.00, 0.0, 0.0, 6.20), FuelClasses(1600.0, 109.0, 30.0, 1500.0, 1400.0), _H8, 2.4, 0.40)
"""SB40 SH4 (144): Low load, humid climate timber-shrub"""
const SH4 = RothermelFuel("SB40 SH4: Low load, humid climate timber-shrub",
    FuelClasses(0.85, 1.15, 0.20, 0.0, 2.55), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 3.0, 0.30)
"""SB40 SH5 (145): High load, dry climate shrub"""
const SH5 = RothermelFuel("SB40 SH5: High load, dry climate shrub",
    FuelClasses(3.60, 2.10, 0.0, 0.0, 2.90), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 6.0, 0.15)
"""SB40 SH6 (146): Low load, humid climate shrub"""
const SH6 = RothermelFuel("SB40 SH6: Low load, humid climate shrub",
    FuelClasses(2.90, 1.45, 0.0, 0.0, 1.40), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 2.0, 0.30)
"""SB40 SH7 (147): Very high load, dry climate shrub"""
const SH7 = RothermelFuel("SB40 SH7: Very high load, dry climate shrub",
    FuelClasses(3.50, 5.30, 2.20, 0.0, 3.40), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 6.0, 0.15)
"""SB40 SH8 (148): High load, humid climate shrub"""
const SH8 = RothermelFuel("SB40 SH8: High load, humid climate shrub",
    FuelClasses(2.05, 3.40, 0.85, 0.0, 4.35), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 3.0, 0.40)
"""SB40 SH9 (149): Very high load, humid climate shrub"""
const SH9 = RothermelFuel("SB40 SH9: Very high load, humid climate shrub",
    FuelClasses(4.50, 2.45, 0.0, 1.55, 7.00), FuelClasses(750.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 4.4, 0.40)

"""SB40 TU1 (161): Light load, dry climate timber-grass-shrub"""
const TU1 = RothermelFuel("SB40 TU1: Light load, dry climate timber-grass-shrub",
    FuelClasses(0.20, 0.90, 1.50, 0.20, 0.90), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1600.0), _H8, 0.6, 0.20)
"""SB40 TU2 (162): Moderate load, humid climate timber-shrub"""
const TU2 = RothermelFuel("SB40 TU2: Moderate load, humid climate timber-shrub",
    FuelClasses(0.95, 1.80, 1.25, 0.0, 0.20), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 1.0, 0.30)
"""SB40 TU3 (163): Moderate load, humid climate timber-grass-shrub"""
const TU3 = RothermelFuel("SB40 TU3: Moderate load, humid climate timber-grass-shrub",
    FuelClasses(1.10, 0.15, 0.25, 0.65, 1.10), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1400.0), _H8, 1.3, 0.30)
"""SB40 TU4 (164): Dwarf conifer understory"""
const TU4 = RothermelFuel("SB40 TU4: Dwarf conifer understory",
    FuelClasses(4.50, 0.0, 0.0, 0.0, 2.00), FuelClasses(2300.0, 109.0, 30.0, 1500.0, 2000.0), _H8, 0.5, 0.12)
"""SB40 TU5 (165): Very high load, dry climate timber-shrub"""
const TU5 = RothermelFuel("SB40 TU5: Very high load, dry climate timber-shrub",
    FuelClasses(4.00, 4.00, 3.00, 0.0, 3.00), FuelClasses(1500.0, 109.0, 30.0, 1500.0, 750.0), _H8, 1.0, 0.25)

"""SB40 TL1 (181): Low load, compact conifer litter"""
const TL1 = RothermelFuel("SB40 TL1: Low load, compact conifer litter",
    FuelClasses(1.00, 2.20, 3.60, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.2, 0.30)
"""SB40 TL2 (182): Low broadleaf litter"""
const TL2 = RothermelFuel("SB40 TL2: Low broadleaf litter",
    FuelClasses(1.40, 2.30, 2.20, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.2, 0.25)
"""SB40 TL3 (183): Moderate load conifer litter"""
const TL3 = RothermelFuel("SB40 TL3: Moderate load conifer litter",
    FuelClasses(0.50, 2.20, 2.80, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.20)
"""SB40 TL4 (184): Small downed logs"""
const TL4 = RothermelFuel("SB40 TL4: Small downed logs",
    FuelClasses(0.50, 1.50, 4.20, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.4, 0.25)
"""SB40 TL5 (185): High load conifer litter"""
const TL5 = RothermelFuel("SB40 TL5: High load conifer litter",
    FuelClasses(1.15, 2.50, 4.40, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.6, 0.25)
"""SB40 TL6 (186): Moderate load broadleaf litter"""
const TL6 = RothermelFuel("SB40 TL6: Moderate load broadleaf litter",
    FuelClasses(2.40, 1.20, 1.20, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.25)
"""SB40 TL7 (187): Large downed logs"""
const TL7 = RothermelFuel("SB40 TL7: Large downed logs",
    FuelClasses(0.30, 1.40, 8.10, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.4, 0.25)
"""SB40 TL8 (188): Long-needle litter"""
const TL8 = RothermelFuel("SB40 TL8: Long-needle litter",
    FuelClasses(5.80, 1.40, 1.10, 0.0, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.35)
"""SB40 TL9 (189): Very high load broadleaf litter"""
const TL9 = RothermelFuel("SB40 TL9: Very high load broadleaf litter",
    FuelClasses(6.65, 3.30, 4.15, 0.0, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.6, 0.35)

"""SB40 SB1 (201): Low activity fuel"""
const SB1 = RothermelFuel("SB40 SB1: Low activity fuel",
    FuelClasses(1.50, 3.00, 11.00, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.0, 0.25)
"""SB40 SB2 (202): Moderate activity fuel or low load blowdown"""
const SB2 = RothermelFuel("SB40 SB2: Moderate activity fuel or low load blowdown",
    FuelClasses(4.50, 4.25, 4.00, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.0, 0.25)
"""SB40 SB3 (203): High activity fuel or moderate load blowdown"""
const SB3 = RothermelFuel("SB40 SB3: High activity fuel or moderate load blowdown",
    FuelClasses(5.50, 2.75, 3.00, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.2, 0.25)
"""SB40 SB4 (204): High load blowdown"""
const SB4 = RothermelFuel("SB40 SB4: High load blowdown",
    FuelClasses(5.25, 3.50, 5.25, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 2.7, 0.25)

# ============================================================================ #
#         McArthur Mk5 Grassland (Noble et al. 1980)
# ============================================================================ #

#-----------------------------------------------------------------------------# McArthurGrasslandROS
"""
    McArthurGrasslandROS()

McArthur Mk5 grassland fire danger meter, expressed as equations by
Noble, Bary & Gill (1980).

# Rate of Spread
    rate_of_spread(model::McArthurGrasslandROS; wind, temperature, humidity, curing, fuel_load) -> Float64  [m/min]

- `wind` — 10-m open wind speed [km/h]
- `temperature` — Air temperature [°C]
- `humidity` — Relative humidity [%]
- `curing` — Grass curing [%] (0 = green, 100 = fully cured)
- `fuel_load` — Fuel load [t/ha] (default: 4.5)

# References
- Noble, I.R., Bary, G.A.V. & Gill, A.M. (1980). McArthur's fire-danger
  meters expressed as equations. Australian Journal of Ecology, 5, 201–203.

### Examples
```julia
model = McArthurGrasslandROS()
rate_of_spread(model; wind=20.0, temperature=35.0, humidity=15.0, curing=90.0)
```
"""
struct McArthurGrasslandROS <: AbstractRateOfSpread end

function rate_of_spread(::McArthurGrasslandROS; wind, temperature, humidity, curing, fuel_load=nothing)
    T = promote_type(typeof(wind), typeof(temperature), typeof(humidity), typeof(curing))
    W_val = fuel_load === nothing ? T(4.5) : T(fuel_load)
    C = T(curing)
    Temp = T(temperature)
    RH = T(humidity)
    U = T(wind)
    W = W_val

    # Fuel moisture content (%)
    m = ((T(97.7) + T(4.06) * RH) / (Temp + T(6))) - T(0.00854) * RH + T(3000) / C - T(30)

    # Grassland Fire Danger Index
    if m >= T(30)
        GFDI = zero(T)
    elseif m < T(18.8)
        GFDI = T(3.35) * W * exp(T(-0.0897) * m + T(0.0403) * U)
    else
        GFDI = T(0.299) * W * exp(T(-1.686) + T(0.0403) * U) * (T(30) - m)
    end

    # ROS: 0.13 * GFDI [km/h] → m/min
    return T(0.13) * GFDI * T(1000) / T(60)
end

# ============================================================================ #
#         McArthur Mk5 Forest (Noble et al. 1980)
# ============================================================================ #

#-----------------------------------------------------------------------------# McArthurForestROS
"""
    McArthurForestROS()

McArthur Mk5 Forest Fire Danger Index, expressed as equations by
Noble, Bary & Gill (1980).

# Rate of Spread
    rate_of_spread(model::McArthurForestROS; wind, temperature, humidity, drought_factor, fuel_load) -> Float64  [m/min]

- `wind` — 10-m open wind speed [km/h]
- `temperature` — Air temperature [°C]
- `humidity` — Relative humidity [%]
- `drought_factor` — Drought factor [0–10]
- `fuel_load` — Surface fuel load [t/ha] (default: 12.5)

# References
- Noble, I.R., Bary, G.A.V. & Gill, A.M. (1980). McArthur's fire-danger
  meters expressed as equations. Australian Journal of Ecology, 5, 201–203.

### Examples
```julia
model = McArthurForestROS()
rate_of_spread(model; wind=25.0, temperature=35.0, humidity=10.0, drought_factor=8.0)
```
"""
struct McArthurForestROS <: AbstractRateOfSpread end

function rate_of_spread(::McArthurForestROS; wind, temperature, humidity, drought_factor, fuel_load=nothing)
    T = promote_type(typeof(wind), typeof(temperature), typeof(humidity), typeof(drought_factor))
    Temp = T(temperature)
    RH = T(humidity)
    U = T(wind)
    DF = T(drought_factor)
    W = fuel_load === nothing ? T(12.5) : T(fuel_load)

    # Forest Fire Danger Index (Noble et al. 1980)
    FFDI = T(2) * exp(T(-0.450) + T(0.987) * log(DF) - T(0.0345) * RH + T(0.0338) * Temp + T(0.0234) * U)

    # ROS: 0.0012 * FFDI * W [km/h] → m/min
    return T(0.0012) * FFDI * W * T(1000) / T(60)
end

# ============================================================================ #
#         Canadian FBP System (Forestry Canada 1992)
# ============================================================================ #

#-----------------------------------------------------------------------------# FBPFuelType
"""
    FBPFuelType{T}(; name, a, b, c, BUI0, Q)

Canadian FBP fuel type parameters for the ISI-based rate of spread formula.

The base ROS is `RSI = a * (1 - exp(-b * ISI))^c` [m/min], adjusted by the
Buildup Effect: `ROS = RSI * exp(50 * ln(Q) * (1/BUI - 1/BUI0))`.

# Fields
- `name::String` — Fuel type code and description
- `a::T` — Maximum ROS parameter [m/min]
- `b::T` — ISI sensitivity parameter
- `c::T` — Shape parameter
- `BUI0::T` — Average BUI for the fuel type
- `Q::T` — Buildup effect parameter (1.0 = no BUI effect)
"""
struct FBPFuelType{T}
    name::String
    a::T
    b::T
    c::T
    BUI0::T
    Q::T
end

function FBPFuelType(name::AbstractString, a, b, c, BUI0, Q)
    T = promote_type(typeof(a), typeof(b), typeof(c), typeof(BUI0), typeof(Q))
    FBPFuelType{T}(String(name), T(a), T(b), T(c), T(BUI0), T(Q))
end

function Base.show(io::IO, f::FBPFuelType{T}) where {T}
    print(io, "FBPFuelType{$T}(\"", f.name, "\")")
end

const FBP_C1  = FBPFuelType("C-1: Spruce-Lichen Woodland",        90.0, 0.0649, 4.5, 72.0, 0.90)
const FBP_C2  = FBPFuelType("C-2: Boreal Spruce",                110.0, 0.0282, 1.5, 64.0, 0.70)
const FBP_C3  = FBPFuelType("C-3: Mature Jack/Lodgepole Pine",   110.0, 0.0444, 3.0, 62.0, 0.75)
const FBP_C4  = FBPFuelType("C-4: Immature Jack/Lodgepole Pine", 110.0, 0.0293, 1.5, 66.0, 0.80)
const FBP_C5  = FBPFuelType("C-5: Red and White Pine",            30.0, 0.0697, 4.0, 56.0, 0.80)
const FBP_C6  = FBPFuelType("C-6: Conifer Plantation",            30.0, 0.0800, 3.0, 62.0, 0.80)
const FBP_C7  = FBPFuelType("C-7: Ponderosa Pine/Douglas Fir",    45.0, 0.0305, 2.0, 106.0, 0.85)
const FBP_D1  = FBPFuelType("D-1: Leafless Aspen",                30.0, 0.0232, 1.6, 32.0, 0.90)
const FBP_M3  = FBPFuelType("M-3: Dead Balsam Fir Mixedwood",   120.0, 0.0572, 1.4, 50.0, 0.80)
const FBP_M4  = FBPFuelType("M-4: Dead Balsam Fir Mixedwood (green)", 100.0, 0.0404, 1.48, 50.0, 0.80)
const FBP_S1  = FBPFuelType("S-1: Jack/Lodgepole Pine Slash",     75.0, 0.0297, 1.3, 38.0, 0.75)
const FBP_S2  = FBPFuelType("S-2: White Spruce/Balsam Slash",     40.0, 0.0438, 1.7, 63.0, 0.75)
const FBP_S3  = FBPFuelType("S-3: Coastal Cedar/Hemlock/Fir Slash", 55.0, 0.0829, 3.2, 31.0, 0.75)
const FBP_O1a = FBPFuelType("O-1a: Matted Grass",                190.0, 0.0310, 1.4, 1.0, 1.00)
const FBP_O1b = FBPFuelType("O-1b: Standing Grass",              250.0, 0.0350, 1.7, 1.0, 1.00)

#-----------------------------------------------------------------------------# CanadianFBP_ROS
"""
    CanadianFBP_ROS(fuel::FBPFuelType)

Canadian Forest Fire Behavior Prediction (FBP) System rate of spread model.

Uses the ISI-based formula with buildup effect correction.

# Rate of Spread
    rate_of_spread(model::CanadianFBP_ROS; ffmc, wind, bui=model.fuel.BUI0, curing=nothing) -> Float64  [m/min]

- `ffmc` — Fine Fuel Moisture Code [0–101]
- `wind` — 10-m open wind speed [km/h]
- `bui` — Buildup Index (default: fuel type average `BUI0`)
- `curing` — Grass curing [%] (required for O-1a/O-1b fuel types)

# References
- Forestry Canada Fire Danger Group (1992). Development and Structure of the
  Canadian Forest Fire Behavior Prediction System. Information Report ST-X-3.
- Wotton, Alexander & Taylor (2009). Updates and revisions to the 1992
  Canadian forest fire behavior prediction system.

### Examples
```julia
model = CanadianFBP_ROS(FBP_C2)
rate_of_spread(model; ffmc=92.0, wind=20.0)
```
"""
struct CanadianFBP_ROS{T} <: AbstractRateOfSpread
    fuel::FBPFuelType{T}
end

function rate_of_spread(model::CanadianFBP_ROS{T}; ffmc, wind, bui=model.fuel.BUI0, curing=nothing) where {T}
    fuel = model.fuel
    FFMC = T(ffmc)
    WS = T(wind)
    BUI = T(bui)
    o = one(T)

    # Fuel moisture from FFMC
    fm = T(147.27723) * (T(101) - FFMC) / (T(59.5) + FFMC)

    # Fuel moisture function
    fF = T(91.9) * exp(T(-0.1386) * fm) * (o + fm^T(5.31) / T(4.93e7))

    # Wind function (with FBP high-wind correction)
    if WS < T(40)
        fW = exp(T(0.05039) * WS)
    else
        fW = T(12) * (o - exp(T(-0.0818) * (WS - T(28))))
    end

    # Initial Spread Index
    ISI = T(0.208) * fW * fF

    # Base rate of spread
    RSI = fuel.a * (o - exp(-fuel.b * ISI))^fuel.c

    # Curing factor (grass fuel types only)
    if curing !== nothing
        CC = T(curing)
        if CC < T(58.8)
            CF = T(0.005) * (exp(T(0.061) * CC) - o)
        else
            CF = T(0.176) + T(0.02) * (CC - T(58.8))
        end
        RSI *= CF
    end

    # Buildup Effect
    if fuel.Q < o && BUI > zero(T)
        BE = exp(T(50) * log(fuel.Q) * (o / BUI - o / fuel.BUI0))
    else
        BE = o
    end

    return RSI * BE
end

end # module
