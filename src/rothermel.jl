#-----------------------------------------------------------------------------# FuelClasses
"""
    FuelClasses(d1, d10, d100, herb, wood)
    FuelClasses(dead, live)

One value per Rothermel fuel class: dead 1-, 10-, and 100-hour timelag classes
(particle diameter under 1/4", 1/4–1", 1–3") and live herbaceous and woody
fuel. Holds per-class fuel parameters in [`FuelModel`](@ref) and moisture
content (fraction of oven-dry weight) in [`spread_rate`](@ref).

The two-argument form assigns `dead` to the three dead classes and `live` to
both live classes.

### Examples
```julia
FuelClasses(0.06, 0.07, 0.08, 0.6, 0.9)
FuelClasses(0.08, 1.0)   # 8% dead, 100% live
```
"""
struct FuelClasses{T}
    d1::T
    d10::T
    d100::T
    herb::T
    wood::T
end
FuelClasses(d1, d10, d100, herb, wood) = FuelClasses(promote(d1, d10, d100, herb, wood)...)
FuelClasses(dead, live) = FuelClasses(dead, dead, dead, live, live)
FuelClasses{T}(c::FuelClasses) where {T} = FuelClasses(T(c.d1), T(c.d10), T(c.d100), T(c.herb), T(c.wood))

Base.map(f, a::FuelClasses) = FuelClasses(f(a.d1), f(a.d10), f(a.d100), f(a.herb), f(a.wood))
Base.map(f, a::FuelClasses, b::FuelClasses) =
    FuelClasses(f(a.d1, b.d1), f(a.d10, b.d10), f(a.d100, b.d100), f(a.herb, b.herb), f(a.wood, b.wood))
Base.map(f, a::FuelClasses, b::FuelClasses, c::FuelClasses) =
    FuelClasses(f(a.d1, b.d1, c.d1), f(a.d10, b.d10, c.d10), f(a.d100, b.d100, c.d100),
                f(a.herb, b.herb, c.herb), f(a.wood, b.wood, c.wood))
sum_dead(c::FuelClasses) = c.d1 + c.d10 + c.d100
sum_live(c::FuelClasses) = c.herb + c.wood

#-----------------------------------------------------------------------------# FuelModel
"""
    FuelModel(code, w, σ, h, δ, Mx)

Rothermel (1972) fuel model: per-class oven-dry load `w` [tons/acre],
surface-area-to-volume ratio `σ` [1/ft], and heat content `h` [BTU/lb] as
[`FuelClasses`](@ref); fuel bed depth `δ` [ft]; dead fuel moisture of
extinction `Mx` [fraction]. `code` is the integer identifier fuel rasters use
(NFFL 1–13, Scott & Burgan 101–204).

Parameters keep the US customary units of the published tables so entries can
be checked against them; [`spread_rate`](@ref) is SI at its interface. The
standard models are `Wildfires.NFFL` (Anderson 1982) and `Wildfires.SB40`
(Scott & Burgan 2005).

### Examples
```julia
Wildfires.NFFL.SHORT_GRASS
FuelModel{Float32}(Wildfires.SB40.GR2)   # for a Float32 grid
```
"""
struct FuelModel{T}
    code::Int
    w::FuelClasses{T}
    σ::FuelClasses{T}
    h::FuelClasses{T}
    δ::T
    Mx::T
end
FuelModel{T}(m::FuelModel) where {T} =
    FuelModel(m.code, FuelClasses{T}(m.w), FuelClasses{T}(m.σ), FuelClasses{T}(m.h), T(m.δ), T(m.Mx))

#-----------------------------------------------------------------------------# FuelBed
"""
    FuelBed(fuel::FuelModel)

Fuel-bed properties of `fuel` that do not depend on moisture, wind, or slope,
computed once so [`spread_rate`](@ref) evaluates only the environment-dependent
terms. Internal units are Rothermel's (ft, lb, BTU, min).

### Examples
```julia
FuelBed(Wildfires.NFFL.SHORT_GRASS)
FuelBed{Float32}(Wildfires.NFFL.SHORT_GRASS)
```
"""
struct FuelBed{T}
    fuel::FuelModel{T}
    f::FuelClasses{T}       # surface-area fraction of each class within its category
    f_dead::T               # dead share of total surface area
    f_live::T               # live share of total surface area
    ε::FuelClasses{T}       # effective heating number exp(-138/σ)
    wε::FuelClasses{T}      # load × ε: fine-fuel weights for the live moisture of extinction
    W::T                    # fine dead / fine live load ratio
    wn_dead::T              # net (mineral-free) load per category [lb/ft²]
    wn_live::T
    h_dead::T               # area-weighted heat content per category [BTU/lb]
    h_live::T
    ρ_b::T                  # bulk density [lb/ft³]
    Γ::T                    # reaction velocity [1/min]
    ξ::T                    # propagating flux ratio
    C_w::T                  # wind factor φ_w = C_w U^B, U in ft/min
    B::T
    C_s::T                  # slope factor φ_s = C_s tan²θ
end
FuelBed{T}(m::FuelModel) where {T} = FuelBed(FuelModel{T}(m))
FuelBed{T}(bed::FuelBed) where {T} = FuelBed(FuelModel{T}(bed.fuel))
FuelBed{T}(bed::FuelBed{T}) where {T} = bed

# Replace a non-positive divisor or power base by one, so absent classes and
# empty fuel beds give finite intermediates instead of Inf/NaN that AD would
# propagate even through a masking `ifelse`.
_pos(x) = ifelse(x > zero(x), x, one(x))

# Albini (1976) size bins for net-load weighting.
_bin(σ::T) where {T} = (σ ≥ T(16)) + (σ ≥ T(48)) + (σ ≥ T(96)) + (σ ≥ T(192)) + (σ ≥ T(1200))
_same_bin(a, b) = _bin(a) == _bin(b)

function FuelBed(fuel::FuelModel{T}) where {T}
    (; σ, h, δ, Mx) = fuel
    z, o = zero(T), one(T)
    ρ_p = T(32)                                  # oven-dry particle density [lb/ft³]
    S_T = T(0.0555)                              # total mineral content
    w = map(x -> x * T(2000 / 43560), fuel.w)    # tons/acre → lb/ft²
    σ⁺ = map(_pos, σ)                            # nonburnable entries have σ = 0

    # Surface area per unit bed area, per class, and the area-fraction weights.
    a = map((σ, w) -> σ * w / ρ_p, σ, w)
    a_dead, a_live = sum_dead(a), sum_live(a)
    f = FuelClasses(a.d1 / _pos(a_dead), a.d10 / _pos(a_dead), a.d100 / _pos(a_dead),
                    a.herb / _pos(a_live), a.wood / _pos(a_live))
    f_dead = a_dead / _pos(a_dead + a_live)
    f_live = a_live / _pos(a_dead + a_live)
    fσ = map(*, f, σ)
    σ_tot = _pos(f_dead * sum_dead(fσ) + f_live * sum_live(fσ))

    # Net load per category. Classes whose σ fall in the same size bin share
    # their summed area fraction (Albini 1976), so a category split across
    # several classes in one bin is not under-weighted. Only categories with
    # both live classes are affected among the standard models.
    g = FuelClasses(f.d1 + f.d10 * _same_bin(σ.d10, σ.d1) + f.d100 * _same_bin(σ.d100, σ.d1),
                    f.d10 + f.d1 * _same_bin(σ.d1, σ.d10) + f.d100 * _same_bin(σ.d100, σ.d10),
                    f.d100 + f.d1 * _same_bin(σ.d1, σ.d100) + f.d10 * _same_bin(σ.d10, σ.d100),
                    f.herb + f.wood * _same_bin(σ.wood, σ.herb),
                    f.wood + f.herb * _same_bin(σ.herb, σ.wood))
    wn = map((g, w) -> g * w * (o - S_T), g, w)
    wn_dead, wn_live = sum_dead(wn), sum_live(wn)
    fh = map(*, f, h)
    h_dead, h_live = sum_dead(fh), sum_live(fh)

    # Fine-fuel weights for the live moisture of extinction (Rothermel eq. 88).
    ε = map(σ -> exp(T(-138) / σ), σ⁺)
    wε = map(*, w, ε)
    W = sum_dead(wε) / _pos(sum_live(map((w, σ) -> w * exp(T(-500) / σ), w, σ⁺)))

    # Packing ratio, reaction velocity, propagating flux, wind and slope coefficients.
    ρ_b = (sum_dead(w) + sum_live(w)) / _pos(δ)
    β = ρ_b / ρ_p
    β_op = T(3.348) * σ_tot^T(-0.8189)
    rpr = β / β_op
    A = T(133) * σ_tot^T(-0.7913)
    Γ_max = σ_tot^T(1.5) / (T(495) + T(0.0594) * σ_tot^T(1.5))
    Γ = Γ_max * (rpr * exp(o - rpr))^A
    ξ = exp((T(0.792) + T(0.681) * sqrt(σ_tot)) * (β + T(0.1))) / (T(192) + T(0.2595) * σ_tot)
    C = T(7.47) * exp(T(-0.133) * σ_tot^T(0.55))
    B = T(0.02526) * σ_tot^T(0.54)
    E = T(0.715) * exp(T(-3.59e-4) * σ_tot)
    C_w = C * _pos(rpr)^(-E)
    C_s = T(5.275) * _pos(β)^T(-0.3)

    FuelBed(fuel, f, f_dead, f_live, ε, wε, W, wn_dead, wn_live, h_dead, h_live, ρ_b, Γ, ξ, C_w, B, C_s)
end

#-----------------------------------------------------------------------------# spread_rate
"""
    spread_rate(bed::FuelBed, moisture::FuelClasses, wind, slope)

Rothermel (1972) head-fire rate of spread [m/s] on `bed` with per-class fuel
`moisture` (fraction of oven-dry weight), midflame `wind` [m/s], and `slope`
as rise over run. Zero at or above the moisture of extinction and for a fuel
bed with no load.

Follows Rothermel (1972) with Albini's (1976) net-load weighting, as in
Andrews (2018). The effective wind-speed limit is not applied (Andrews et al.
2013), so spread grows as `wind^B` without bound. `wind` is the midflame wind
with no adjustment from a reference height. Negative `wind` is treated as calm
and negative `slope` as flat, matching WRF-SFIRE, so the arguments can be the
wind and slope components along the spread direction.

### Examples
```julia
bed = FuelBed(Wildfires.NFFL.SHORT_GRASS)
spread_rate(bed, FuelClasses(0.08, 1.0), 2.2352, 0.0)   # 5 mi/h midflame wind
```
"""
function spread_rate(bed::FuelBed{T}, moisture::FuelClasses, wind, slope) where {T}
    (; f, f_dead, f_live, ε, wε, W, wn_dead, wn_live, h_dead, h_live, ρ_b, Γ, ξ, C_w, B, C_s) = bed
    Mx = bed.fuel.Mx
    M = FuelClasses{T}(moisture)
    U = T(wind) * T(3.28084 * 60)     # m/s → ft/min
    s = max(T(slope), zero(T))
    z, o = zero(T), one(T)

    fM = map(*, f, M)
    mf_dead, mf_live = sum_dead(fM), sum_live(fM)

    # Live moisture of extinction from the fine dead fuel moisture (Rothermel eq. 88).
    mfpd = sum_dead(map(*, wε, M)) / _pos(sum_dead(wε))
    Mx_live = max(T(2.9) * W * (o - mfpd / _pos(Mx)) - T(0.226), Mx)

    # Reaction intensity [BTU/ft²/min]; η_s is the mineral damping at S_e = 0.01.
    η_M_dead = _moisture_damping(mf_dead / _pos(Mx))
    η_M_live = _moisture_damping(mf_live / _pos(Mx_live))
    η_s = T(0.174) * T(0.01)^T(-0.19)
    I_R = Γ * (wn_dead * h_dead * η_M_dead + wn_live * h_live * η_M_live) * η_s

    # The clamp keeps d(U^B)/dU finite at U = 0.
    φ_w = C_w * max(U, T(1e-10))^B
    φ_s = C_s * s^2

    # Heat sink [BTU/ft³]: bulk density × effective heating number × heat of preignition.
    fQ = map((f, ε, m) -> f * ε * (T(250) + T(1116) * m), f, ε, M)
    heat_sink = ρ_b * (f_dead * sum_dead(fQ) + f_live * sum_live(fQ))

    R = I_R * ξ * (o + φ_w + φ_s) / _pos(heat_sink)   # ft/min
    return ifelse(heat_sink > z, R * T(0.3048 / 60), z)
end

# Rothermel eq. 29: moisture damping, zero at and above the moisture of extinction.
function _moisture_damping(r::T) where {T}
    η = one(T) - T(2.59) * r + T(5.11) * r^2 - T(3.52) * r^3
    return ifelse(r < one(T), η, zero(T))
end

#-----------------------------------------------------------------------------# Standard fuel models
const _H8 = FuelClasses(8000.0, 8000.0, 8000.0, 8000.0, 8000.0)   # heat content [BTU/lb]
const _H9 = FuelClasses(9000.0, 9000.0, 9000.0, 9000.0, 9000.0)
const _σ_STD = (109.0, 30.0, 1500.0, 1500.0)                      # σ for (d10, d100, herb, wood) [1/ft]

"""
    NFFL

The 13 Northern Forest Fire Laboratory fuel models (Anderson 1982) as a
`NamedTuple` of [`FuelModel`](@ref)s, codes 1–13.
"""
const NFFL = (
    SHORT_GRASS          = FuelModel(1,  FuelClasses(0.74, 0.0, 0.0, 0.0, 0.0),      FuelClasses(3500.0, _σ_STD...), _H8, 1.0, 0.12),
    TIMBER_GRASS         = FuelModel(2,  FuelClasses(2.0, 1.0, 0.5, 0.5, 0.0),       FuelClasses(3000.0, _σ_STD...), _H8, 1.0, 0.15),
    TALL_GRASS           = FuelModel(3,  FuelClasses(3.01, 0.0, 0.0, 0.0, 0.0),      FuelClasses(1500.0, _σ_STD...), _H8, 2.5, 0.25),
    CHAPARRAL            = FuelModel(4,  FuelClasses(5.01, 4.01, 2.0, 0.0, 5.01),    FuelClasses(2000.0, _σ_STD...), _H8, 6.0, 0.20),
    BRUSH                = FuelModel(5,  FuelClasses(1.0, 0.5, 0.0, 0.0, 2.0),       FuelClasses(2000.0, _σ_STD...), _H8, 2.0, 0.20),
    DORMANT_BRUSH        = FuelModel(6,  FuelClasses(1.5, 2.5, 2.0, 0.0, 0.0),       FuelClasses(1750.0, _σ_STD...), _H8, 2.5, 0.25),
    SOUTHERN_ROUGH       = FuelModel(7,  FuelClasses(1.13, 1.87, 1.5, 0.0, 0.37),    FuelClasses(1750.0, _σ_STD...), _H8, 2.5, 0.40),
    CLOSED_TIMBER_LITTER = FuelModel(8,  FuelClasses(1.5, 1.0, 2.5, 0.0, 0.0),       FuelClasses(2000.0, _σ_STD...), _H8, 0.2, 0.30),
    HARDWOOD_LITTER      = FuelModel(9,  FuelClasses(2.92, 0.41, 0.15, 0.0, 0.0),    FuelClasses(2500.0, _σ_STD...), _H8, 0.2, 0.25),
    TIMBER_UNDERSTORY    = FuelModel(10, FuelClasses(3.01, 2.0, 5.01, 0.0, 2.0),     FuelClasses(2000.0, _σ_STD...), _H8, 1.0, 0.25),
    LIGHT_SLASH          = FuelModel(11, FuelClasses(1.5, 4.51, 5.51, 0.0, 0.0),     FuelClasses(1500.0, _σ_STD...), _H8, 1.0, 0.15),
    MEDIUM_SLASH         = FuelModel(12, FuelClasses(4.01, 14.03, 16.53, 0.0, 0.0),  FuelClasses(1500.0, _σ_STD...), _H8, 2.3, 0.20),
    HEAVY_SLASH          = FuelModel(13, FuelClasses(7.01, 23.04, 28.05, 0.0, 0.0),  FuelClasses(1500.0, _σ_STD...), _H8, 3.0, 0.25),
)

"""
    SB40

The 40 Scott & Burgan (2005) fuel models as a `NamedTuple` of
[`FuelModel`](@ref)s, codes 101–204. Loads are the static values: the dynamic
models (GR, GS, SH9, TU1, TU3) normally transfer cured herbaceous load from
live to dead, which is not implemented, so cured grass spreads slower here
than in BehavePlus.
"""
const SB40 = (
    # Grass
    GR1 = FuelModel(101, FuelClasses(0.10, 0.0, 0.0, 0.30, 0.0),    FuelClasses(2200.0, 109.0, 30.0, 2000.0, 1500.0), _H8, 0.4, 0.15),
    GR2 = FuelModel(102, FuelClasses(0.10, 0.0, 0.0, 1.00, 0.0),    FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 1.0, 0.15),
    GR3 = FuelModel(103, FuelClasses(0.10, 0.40, 0.0, 1.50, 0.0),   FuelClasses(1500.0, 109.0, 30.0, 1300.0, 1500.0), _H8, 2.0, 0.30),
    GR4 = FuelModel(104, FuelClasses(0.25, 0.0, 0.0, 1.90, 0.0),    FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 2.0, 0.15),
    GR5 = FuelModel(105, FuelClasses(0.40, 0.0, 0.0, 2.50, 0.0),    FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1500.0), _H8, 1.5, 0.40),
    GR6 = FuelModel(106, FuelClasses(0.10, 0.0, 0.0, 3.40, 0.0),    FuelClasses(2200.0, 109.0, 30.0, 2000.0, 1500.0), _H9, 1.5, 0.40),
    GR7 = FuelModel(107, FuelClasses(1.00, 0.0, 0.0, 5.40, 0.0),    FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 3.0, 0.15),
    GR8 = FuelModel(108, FuelClasses(0.50, 1.00, 0.0, 7.30, 0.0),   FuelClasses(1500.0, 109.0, 30.0, 1300.0, 1500.0), _H8, 4.0, 0.30),
    GR9 = FuelModel(109, FuelClasses(1.00, 1.00, 0.0, 9.00, 0.0),   FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1500.0), _H8, 5.0, 0.40),
    # Grass-shrub
    GS1 = FuelModel(121, FuelClasses(0.20, 0.0, 0.0, 0.50, 0.65),   FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1800.0), _H8, 0.9, 0.15),
    GS2 = FuelModel(122, FuelClasses(0.50, 0.50, 0.0, 0.60, 1.00),  FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1800.0), _H8, 1.5, 0.15),
    GS3 = FuelModel(123, FuelClasses(0.30, 0.25, 0.0, 1.45, 1.25),  FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1600.0), _H8, 1.8, 0.40),
    GS4 = FuelModel(124, FuelClasses(1.90, 0.30, 0.10, 3.40, 7.10), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1600.0), _H8, 2.1, 0.40),
    # Shrub
    SH1 = FuelModel(141, FuelClasses(0.25, 0.25, 0.0, 0.15, 1.30),  FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1600.0), _H8, 1.0, 0.15),
    SH2 = FuelModel(142, FuelClasses(1.35, 2.40, 0.75, 0.0, 3.85),  FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 1.0, 0.15),
    SH3 = FuelModel(143, FuelClasses(0.45, 3.00, 0.0, 0.0, 6.20),   FuelClasses(1600.0, 109.0, 30.0, 1500.0, 1400.0), _H8, 2.4, 0.40),
    SH4 = FuelModel(144, FuelClasses(0.85, 1.15, 0.20, 0.0, 2.55),  FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 3.0, 0.30),
    SH5 = FuelModel(145, FuelClasses(3.60, 2.10, 0.0, 0.0, 2.90),   FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0),  _H8, 6.0, 0.15),
    SH6 = FuelModel(146, FuelClasses(2.90, 1.45, 0.0, 0.0, 1.40),   FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0),  _H8, 2.0, 0.30),
    SH7 = FuelModel(147, FuelClasses(3.50, 5.30, 2.20, 0.0, 3.40),  FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0),  _H8, 6.0, 0.15),
    SH8 = FuelModel(148, FuelClasses(2.05, 3.40, 0.85, 0.0, 4.35),  FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0),  _H8, 3.0, 0.40),
    SH9 = FuelModel(149, FuelClasses(4.50, 2.45, 0.0, 1.55, 7.00),  FuelClasses(750.0, 109.0, 30.0, 1800.0, 1500.0),  _H8, 4.4, 0.40),
    # Timber-understory
    TU1 = FuelModel(161, FuelClasses(0.20, 0.90, 1.50, 0.20, 0.90), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1600.0), _H8, 0.6, 0.20),
    TU2 = FuelModel(162, FuelClasses(0.95, 1.80, 1.25, 0.0, 0.20),  FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 1.0, 0.30),
    TU3 = FuelModel(163, FuelClasses(1.10, 0.15, 0.25, 0.65, 1.10), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1400.0), _H8, 1.3, 0.30),
    TU4 = FuelModel(164, FuelClasses(4.50, 0.0, 0.0, 0.0, 2.00),    FuelClasses(2300.0, 109.0, 30.0, 1500.0, 2000.0), _H8, 0.5, 0.12),
    TU5 = FuelModel(165, FuelClasses(4.00, 4.00, 3.00, 0.0, 3.00),  FuelClasses(1500.0, 109.0, 30.0, 1500.0, 750.0),  _H8, 1.0, 0.25),
    # Timber litter
    TL1 = FuelModel(181, FuelClasses(1.00, 2.20, 3.60, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.2, 0.30),
    TL2 = FuelModel(182, FuelClasses(1.40, 2.30, 2.20, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.2, 0.25),
    TL3 = FuelModel(183, FuelClasses(0.50, 2.20, 2.80, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.20),
    TL4 = FuelModel(184, FuelClasses(0.50, 1.50, 4.20, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.4, 0.25),
    TL5 = FuelModel(185, FuelClasses(1.15, 2.50, 4.40, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.6, 0.25),
    TL6 = FuelModel(186, FuelClasses(2.40, 1.20, 1.20, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.25),
    TL7 = FuelModel(187, FuelClasses(0.30, 1.40, 8.10, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.4, 0.25),
    TL8 = FuelModel(188, FuelClasses(5.80, 1.40, 1.10, 0.0, 0.0),   FuelClasses(1800.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.35),
    TL9 = FuelModel(189, FuelClasses(6.65, 3.30, 4.15, 0.0, 0.0),   FuelClasses(1800.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.6, 0.35),
    # Slash-blowdown
    SB1 = FuelModel(201, FuelClasses(1.50, 3.00, 11.00, 0.0, 0.0),  FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.0, 0.25),
    SB2 = FuelModel(202, FuelClasses(4.50, 4.25, 4.00, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.0, 0.25),
    SB3 = FuelModel(203, FuelClasses(5.50, 2.75, 3.00, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.2, 0.25),
    SB4 = FuelModel(204, FuelClasses(5.25, 3.50, 5.25, 0.0, 0.0),   FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 2.7, 0.25),
)
