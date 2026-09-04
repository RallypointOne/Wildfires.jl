module Rothermel

#------------------------------------------------------------------------------# FuelClasses
# dead fuel note: after 1/10/100 hrs, particle has closed 1 - 1/e (~63%) of moisture gap with EMC
struct FuelClasses{T}
    d1::T       # dead 1-hr moisture (<1/4")
    d10::T      # dead 10-hr moisture (1/4-1")
    d100::T     # dead 100-hr moisture (1-3")
    herb::T     # live herbaceous
    wood::T     # live woody
end

Base.eltype(::Type{FuelClasses{T}}) where {T} = T
Base.map(f, a::FuelClasses) = FuelClasses(f(a.d1), f(a.d10), f(a.d100), f(a.herb), f(a.wood))
Base.map(f, a::FuelClasses, b::FuelClasses) = FuelClasses(f(a.d1, b.d1), f(a.d10, b.d10), f(a.d100, b.d100), f(a.herb, b.herb), f(a.wood, b.wood))
Base.sum(f::Function, fc::FuelClasses) = f(fc.d1) + f(fc.d10) + f(fc.d100) + f(fc.herb) + f(fc.wood)

#------------------------------------------------------------------------------# FuelModel
struct FuelModel{T}
    code::Int
    w::FuelClasses{T}   # w = Oven-dry fuel load [tons/acre] — how much fuel is present
    σ::FuelClasses{T}   # σ = Surface-area-to-volume ratio [1/ft] — fine fuels (high σ) ignite more easily
    h::FuelClasses{T}   # h = Heat content [BTU/lb] — energy released per unit mass when burned
    δ::T                # δ = Fuel bed depth [ft] — determines packing ratio
    Mx::T               # Mx = Dead fuel moisture of extinction [fraction] — moisture above which fire won't spread
    FuelModel(code::Int, w::FuelClasses{T}, σ::FuelClasses{T}, h::FuelClasses{T}, δ::T, Mx::T) where {T} = new{T}(code, w, σ, h, δ, Mx)
    FuelModel(code::Int=0; w::FuelClasses{T}, σ::FuelClasses{T}, h::FuelClasses{T}, δ::T, Mx::T) where {T} = new{T}(code, w, σ, h, δ, Mx)
end

# Aerodynamic roughness length [m], estimated as δ/10 (Albini 1979).  Often denoted "z0"
roughness_length(fm::FuelModel) = fm.δ * 0.3048 / 10

#------------------------------------------------------------------------------# FuelBed
# The fuel bed assembled from a FuelModel: bulk properties that depend only on the
# fuel, not on moisture/wind/slope, so they are computed once and reused per cell.
# w_i    = fuel load per unit area [lb/ft²], per class
# f      = surface-area fraction within each life category [dimensionless]
# a_dead = total dead surface area per unit bed area [ft²/ft²]
# a_live = total live surface area per unit bed area [ft²/ft²]
# a_tot  = a_dead + a_live [ft²/ft²]
# f_dead = dead fraction of total surface area [dimensionless]
# f_live = live fraction of total surface area [dimensionless]
# σ_tot  = characteristic SAV ratio [1/ft]
struct FuelBed{T}
    fuel::FuelModel{T}
    w_i::FuelClasses{T}
    f::FuelClasses{T}
    a_dead::T
    a_live::T
    a_tot::T
    f_dead::T
    f_live::T
    σ_tot::T
end

function FuelBed(fuel::FuelModel{T}) where {T}
    (; w, σ) = fuel
    z = zero(T)
    ρ_P = T(_ρ_P)
    tons_acre = T(_TONS_ACRE_TO_LB_FT2)

    w_i = map(x -> x * tons_acre, w)             # fuel load [lb/ft²]
    a = map(x -> x / ρ_P, map(*, σ, w_i))        # surface area per unit bed area [ft²/ft²], per class
    a_dead = a.d1 + a.d10 + a.d100
    a_live = a.herb + a.wood
    a_tot = a_dead + a_live

    f = FuelClasses(
        a_dead > z ? a.d1 / a_dead : z,
        a_dead > z ? a.d10 / a_dead : z,
        a_dead > z ? a.d100 / a_dead : z,
        a_live > z ? a.herb / a_live : z,
        a_live > z ? a.wood / a_live : z,
    )

    f_dead = a_tot > z ? a_dead / a_tot : z
    f_live = a_tot > z ? a_live / a_tot : z
    σ_dead = f.d1 * σ.d1 + f.d10 * σ.d10 + f.d100 * σ.d100
    σ_live = f.herb * σ.herb + f.wood * σ.wood
    σ_tot = f_dead * σ_dead + f_live * σ_live

    FuelBed(fuel, w_i, f, a_dead, a_live, a_tot, f_dead, f_live, σ_tot)
end

#-----------------------------------------------------------------------------# Constants
const _ρ_P = 32.0                       # Oven-dry particle density [lb/ft³]
const _S_T = 0.0555                     # Total mineral content [fraction]
const _S_E = 0.01                       # Effective mineral content [fraction]
const _TONS_ACRE_TO_LB_FT2 = 2000.0 / 43560.0
const _KMH_TO_FT_MIN = 3280.84 / 60.0  # km/h → ft/min
const _KMH_TO_MPH = 1.0 / 1.60934      # km/h → mi/h
const _FT_TO_M = 0.3048                 # ft → m


#------------------------------------------------------------------------------# length_to_breadth_ratio
# Length-to-breadth ratio of the fire ellipse (Anderson 1983 / Andrews 2018).
# LB = 1 + 0.25 * U, where U is effective midflame wind speed [mi/h].
# Input:  wind_speed [km/h]
# Output: LB [dimensionless, ≥ 1]
function length_to_breadth_ratio(wind_speed::T) where {T}
    U = wind_speed * T(_KMH_TO_MPH)  # mi/h
    return T(1) + T(0.25) * U
end

#------------------------------------------------------------------------------# spread_rate
# Head fire rate of spread [m/min].
# Inputs:  moisture [fraction 0–1], wind_speed [km/h], slope [rise/run]
#          wind_speed is the *midflame* wind, not 10-m or 20-ft wind.  No wind adjustment
#          factor is applied here; see roughness_length for the input to one.
# Output:  R [m/min]
# Internal calculations use US customary units (ft, lb, BTU) per Rothermel (1972);
# the result is converted to metric at the end.
#
# AD compatibility (Enzyme.jl):
#   - The positional-argument method (bed, moisture, wind_speed, slope) is the AD entry
#     point.  Enzyme does not reliably support keyword arguments.
#   - U^B (fractional power, B ≈ 0.54) is clamped via max(U, 1e-10) to avoid infinite
#     derivatives at zero wind speed.
#   - FuelClasses is immutable, so Enzyme's Duplicated cannot accumulate gradients into
#     a shadow.  Use a closure to differentiate w.r.t. individual moisture components:
#       dR = autodiff(Reverse, m -> spread_rate(bed, FuelClasses(m, ...), ws, s), Active, Active(0.05))
spread_rate(fuel::FuelModel{T}; kw...) where {T} = spread_rate(FuelBed(fuel); kw...)
spread_rate(fuel::FuelModel{T}, moisture::FuelClasses{T}, wind_speed::T, slope::T) where {T} = spread_rate(FuelBed(fuel), moisture, wind_speed, slope)
spread_rate(bed::FuelBed{T}; moisture::FuelClasses{T}, wind_speed::T, slope::T) where {T} = spread_rate(bed, moisture, wind_speed, slope)

function spread_rate(bed::FuelBed{T}, moisture::FuelClasses{T}, wind_speed::T, slope::T) where {T}
    (; fuel, w_i, f, a_dead, a_live, a_tot, f_dead, f_live, σ_tot) = bed
    (; σ, h, δ, Mx) = fuel
    M = moisture
    z = zero(T)
    o = one(T)
    δ > z || return z

    ρ_P = T(_ρ_P)
    S_T = T(_S_T)
    S_E = T(_S_E)
    kmh_ft = T(_KMH_TO_FT_MIN)
    ft_m = T(_FT_TO_M)

    a_tot == z && return z

    U = wind_speed * kmh_ft  # midflame wind speed [ft/min]

    # Net fuel load: subtract mineral content (minerals don't combust)
    wn = map(x -> x * (o - S_T), w_i)
    wn_dead = f.d1 * wn.d1 + f.d10 * wn.d10 + f.d100 * wn.d100  # area-weighted net dead load
    wn_live = f.herb * wn.herb + f.wood * wn.wood               # area-weighted net live load

    # Area-weighted mean moisture content for dead and live pools
    mf_dead = f.d1 * M.d1 + f.d10 * M.d10 + f.d100 * M.d100
    mf_live = f.herb * M.herb + f.wood * M.wood

    # Area-weighted mean heat content for dead and live pools
    h_dead = f.d1 * h.d1 + f.d10 * h.d10 + f.d100 * h.d100
    h_live = f.herb * h.herb + f.wood * h.wood

    # Packing ratio β: bulk density / particle density — how tightly the fuel bed is packed.
    # β_op: optimum packing ratio where reaction velocity is maximized (empirical fit to σ_tot).
    # rpr: relative packing ratio — how far actual packing deviates from optimum.
    w_total = sum(x -> x / ρ_P, w_i)
    β = w_total / δ
    β_op = T(3.348) * σ_tot^T(-0.8189)
    rpr = β / β_op

    # Cache exp(-138/σ) values reused across W and heat-sink calculations.
    # The -138 exponent is the Rothermel effective heating number coefficient.
    exp_d1 = exp(T(-138) / σ.d1)
    exp_d10 = exp(T(-138) / σ.d10)
    exp_d100 = exp(T(-138) / σ.d100)
    exp_herb = exp(T(-138) / σ.herb)
    exp_wood = exp(T(-138) / σ.wood)

    if a_live > z
        # Live fuel moisture of extinction (mx_live): the moisture level above which live fuel
        # will not sustain combustion.  It depends on the ratio of fine dead to fine live fuel
        # loading (W) and the weighted dead fuel moisture (mfpd).  Fine-fuel SAV weighting
        # (exp(-138/σ) and exp(-500/σ)) emphasizes the particles most likely to ignite.
        W_num = w_i.d1 * exp_d1 + w_i.d10 * exp_d10 + w_i.d100 * exp_d100
        W_den = w_i.herb * exp(T(-500) / σ.herb) + w_i.wood * exp(T(-500) / σ.wood)
        W = W_den > z ? W_num / W_den : z
        mfpd_num = w_i.d1 * M.d1 * exp_d1 + w_i.d10 * M.d10 * exp_d10 + w_i.d100 * M.d100 * exp_d100
        mfpd = W_num > z ? mfpd_num / W_num : z
        mx_live = T(2.9) * W * (o - mfpd / Mx) - T(0.226)
        mx_live = max(mx_live, Mx)   # can't be lower than dead extinction moisture
    else
        mx_live = Mx
    end

    # Mineral damping coefficient η_s: accounts for silica-free mineral content S_E, which
    # absorbs heat without contributing to combustion.  Because S_E is a fixed constant here
    # this always evaluates to ≈ 0.4174; Rothermel caps it at 1.
    η_s = T(0.174) * S_E^T(-0.19)

    # Moisture damping coefficients η_M: reduce reaction intensity as moisture increases.
    # Polynomial fit to experimental data; goes to 0 when moisture reaches extinction.
    rm_dead = Mx > z ? mf_dead / Mx : z
    η_M_dead = mf_dead >= Mx ? z : o - T(2.59) * rm_dead + T(5.11) * rm_dead^2 - T(3.52) * rm_dead^3

    if a_live > z && mx_live > z
        rm_live = mf_live / mx_live
        η_M_live = mf_live >= mx_live ? z : o - T(2.59) * rm_live + T(5.11) * rm_live^2 - T(3.52) * rm_live^3
    else
        η_M_live = z
    end

    # Reaction velocity Γ [1/min]: rate at which the fuel complex reacts (burns).
    # Γ_max is the theoretical maximum at optimum packing; A shapes the bell curve around β_op.
    A = T(133) * σ_tot^T(-0.7913)
    Γ_max = σ_tot^T(1.5) / (T(495) + T(0.0594) * σ_tot^T(1.5))
    Γ = Γ_max * (rpr * exp(o - rpr))^A

    # Reaction intensity I_R [BTU/ft²/min]: heat released per unit area of fuel bed per minute.
    # Combines reaction velocity with available net fuel energy, damped by moisture and minerals.
    I_R = Γ * (wn_dead * h_dead * η_M_dead + wn_live * h_live * η_M_live) * η_s

    # Propagating flux ratio ξ: fraction of I_R that is transferred ahead of the flame front
    # to preheat and ignite unburned fuel (rather than going up or laterally as convection/radiation).
    ξ = (T(192) + T(0.2595) * σ_tot)^(-o) * exp((T(0.792) + T(0.681) * sqrt(σ_tot)) * (β + T(0.1)))

    # Wind factor φ_w: multiplier that amplifies spread rate with wind speed.
    # C, B, E are empirical coefficients fitted to σ_tot and β.
    # NOTE: the effective wind speed limit (Rothermel 1972; revised by Andrews et al. 2013)
    # is NOT applied.  φ_w grows as U^B with B ≈ 2 for fine fuels, so R is unbounded —
    # NFFL 1 at 80 km/h returns ~3500 m/min.  Cap externally if high winds are in scope.
    C = T(7.47) * exp(T(-0.133) * σ_tot^T(0.55))
    B = T(0.02526) * σ_tot^T(0.54)
    E = T(0.715) * exp(T(-3.59e-4) * σ_tot)
    φ_w = C * max(U, T(1e-10))^B * rpr^(-E)  # clamp avoids ∞ derivative of U^B (B<1) at U=0

    # Slope factor φ_s: amplifies spread rate on uphill slopes (slope = rise/run).
    # Rothermel fits this for upslope only.  slope^2 is symmetric, so a downhill (negative)
    # slope is accelerated exactly like the equivalent uphill.  Pass max(slope, 0) if the
    # caller resolves slope along the spread direction.
    φ_s = T(5.275) * β^T(-0.3) * slope^2

    # Heat sink: energy [BTU/ft²] required to raise unburned fuel ahead of the front to
    # ignition temperature.  ρ_b is bulk density [lb/ft³]; eps (effective heating number)
    # accounts for how deep into the fuel bed the preheating penetrates and moisture load.
    ρ_b = β * ρ_P
    eps = f_dead * (
        f.d1 * (T(250) + T(1116) * M.d1) * exp_d1 +
        f.d10 * (T(250) + T(1116) * M.d10) * exp_d10 +
        f.d100 * (T(250) + T(1116) * M.d100) * exp_d100
    ) +
          f_live * (
        f.herb * (T(250) + T(1116) * M.herb) * exp_herb +
        f.wood * (T(250) + T(1116) * M.wood) * exp_wood
    )

    heat_sink = ρ_b * eps
    heat_sink ≤ z && return z

    # Rate of spread [ft/min] = heat source / heat sink, amplified by wind and slope.
    # Convert to m/min for output.
    R = I_R * ξ * (o + φ_w + φ_s) / heat_sink
    return R * ft_m
end

#------------------------------------------------------------------------------# MODELS
# _H8: standard heat content [BTU/lb] assumed uniform across all classes for NFFL models.
# _σ_STD: standard SAV ratios [1/ft] for d10, d100, herb, wood used by both NFFL and SB40.
# d1 SAV varies per model (finer grass has higher SAV than coarser slash).
const _H8 = FuelClasses(8000.0, 8000.0, 8000.0, 8000.0, 8000.0)  # Standard heat content [BTU/lb]
const _σ_STD = (109.0, 30.0, 1500.0, 1500.0)                    # Standard σ for (d10, d100, herb, wood) [1/ft]

# NFFL fuel models (Anderson, 1982)
const NFFL = (
    SHORT_GRASS=FuelModel(1, FuelClasses(0.74, 0.0, 0.0, 0.0, 0.0), FuelClasses(3500.0, _σ_STD...), _H8, 1.0, 0.12),
    TIMBER_GRASS=FuelModel(2, FuelClasses(2.0, 1.0, 0.5, 0.5, 0.0), FuelClasses(3000.0, _σ_STD...), _H8, 1.0, 0.15),
    TALL_GRASS=FuelModel(3, FuelClasses(3.01, 0.0, 0.0, 0.0, 0.0), FuelClasses(1500.0, _σ_STD...), _H8, 2.5, 0.25),
    CHAPARRAL=FuelModel(4, FuelClasses(5.01, 4.01, 2.0, 0.0, 5.01), FuelClasses(2000.0, _σ_STD...), _H8, 6.0, 0.20),
    BRUSH=FuelModel(5, FuelClasses(1.0, 0.5, 0.0, 0.0, 2.0), FuelClasses(2000.0, _σ_STD...), _H8, 2.0, 0.20),
    DORMANT_BRUSH=FuelModel(6, FuelClasses(1.5, 2.5, 2.0, 0.0, 0.0), FuelClasses(1750.0, _σ_STD...), _H8, 2.5, 0.25),
    SOUTHERN_ROUGH=FuelModel(7, FuelClasses(1.13, 1.87, 1.5, 0.0, 0.37), FuelClasses(1750.0, _σ_STD...), _H8, 2.5, 0.40),
    CLOSED_TIMBER_LITTER=FuelModel(8, FuelClasses(1.5, 1.0, 2.5, 0.0, 0.0), FuelClasses(2000.0, _σ_STD...), _H8, 0.2, 0.30),
    HARDWOOD_LITTER=FuelModel(9, FuelClasses(2.92, 0.41, 0.15, 0.0, 0.0), FuelClasses(2500.0, _σ_STD...), _H8, 0.2, 0.25),
    TIMBER_UNDERSTORY=FuelModel(10, FuelClasses(3.01, 2.0, 5.01, 0.0, 2.0), FuelClasses(2000.0, _σ_STD...), _H8, 1.0, 0.25),
    LIGHT_SLASH=FuelModel(11, FuelClasses(1.5, 4.51, 5.51, 0.0, 0.0), FuelClasses(1500.0, _σ_STD...), _H8, 1.0, 0.15),
    MEDIUM_SLASH=FuelModel(12, FuelClasses(4.01, 14.03, 16.53, 0.0, 0.0), FuelClasses(1500.0, _σ_STD...), _H8, 2.3, 0.20),
    HEAVY_SLASH=FuelModel(13, FuelClasses(7.01, 23.04, 28.05, 0.0, 0.0), FuelClasses(1500.0, _σ_STD...), _H8, 3.0, 0.25),
)

# Scott & Burgan (2005) 40 fuel models (GTR-153/371).
# Loads are the *static* values: the dynamic models (GR, GS, SH9, TU1) normally transfer
# part of the herbaceous load from live to dead as a function of herb moisture (curing).
# That transfer is not implemented, so cured grass spreads slower here than in BehavePlus.
const SB40 = (
    # Grass
    GR1=FuelModel(101, FuelClasses(0.10, 0.0, 0.0, 0.30, 0.0), FuelClasses(2200.0, 109.0, 30.0, 2000.0, 1500.0), _H8, 0.4, 0.15),
    GR2=FuelModel(102, FuelClasses(0.10, 0.0, 0.0, 1.00, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 1.0, 0.15),
    GR3=FuelModel(103, FuelClasses(0.10, 0.40, 0.0, 1.50, 0.0), FuelClasses(1500.0, 109.0, 30.0, 1300.0, 1500.0), _H8, 2.0, 0.30),
    GR4=FuelModel(104, FuelClasses(0.25, 0.0, 0.0, 1.90, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 2.0, 0.15),
    GR5=FuelModel(105, FuelClasses(0.40, 0.0, 0.0, 2.50, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1500.0), _H8, 1.5, 0.40),
    GR6=FuelModel(106, FuelClasses(0.10, 0.0, 0.0, 3.40, 0.0), FuelClasses(2200.0, 109.0, 30.0, 2000.0, 1500.0), _H8, 1.5, 0.40),
    GR7=FuelModel(107, FuelClasses(1.00, 0.0, 0.0, 5.40, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 3.0, 0.15),
    GR8=FuelModel(108, FuelClasses(0.50, 1.00, 0.0, 7.30, 0.0), FuelClasses(1500.0, 109.0, 30.0, 1300.0, 1500.0), _H8, 4.0, 0.30),
    GR9=FuelModel(109, FuelClasses(1.00, 1.00, 0.0, 9.00, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1500.0), _H8, 5.0, 0.40),
    # Grass-Shrub
    GS1=FuelModel(121, FuelClasses(0.20, 0.0, 0.0, 0.50, 0.65), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1800.0), _H8, 0.9, 0.15),
    GS2=FuelModel(122, FuelClasses(0.50, 0.50, 0.0, 0.60, 1.00), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1800.0), _H8, 1.5, 0.15),
    GS3=FuelModel(123, FuelClasses(0.30, 0.25, 0.0, 1.45, 1.25), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1600.0), _H8, 1.8, 0.40),
    GS4=FuelModel(124, FuelClasses(1.90, 0.30, 0.10, 3.40, 7.10), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1600.0), _H8, 2.1, 0.40),
    # Shrub
    SH1=FuelModel(141, FuelClasses(0.25, 0.25, 0.0, 0.15, 1.30), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1600.0), _H8, 1.0, 0.15),
    SH2=FuelModel(142, FuelClasses(1.35, 2.40, 0.75, 0.0, 3.85), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 1.0, 0.15),
    SH3=FuelModel(143, FuelClasses(0.45, 3.00, 0.0, 0.0, 6.20), FuelClasses(1600.0, 109.0, 30.0, 1500.0, 1400.0), _H8, 2.4, 0.40),
    SH4=FuelModel(144, FuelClasses(0.85, 1.15, 0.20, 0.0, 2.55), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 3.0, 0.30),
    SH5=FuelModel(145, FuelClasses(3.60, 2.10, 0.0, 0.0, 2.90), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 6.0, 0.15),
    SH6=FuelModel(146, FuelClasses(2.90, 1.45, 0.0, 0.0, 1.40), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 2.0, 0.30),
    SH7=FuelModel(147, FuelClasses(3.50, 5.30, 2.20, 0.0, 3.40), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 6.0, 0.15),
    SH8=FuelModel(148, FuelClasses(2.05, 3.40, 0.85, 0.0, 4.35), FuelClasses(750.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 3.0, 0.40),
    SH9=FuelModel(149, FuelClasses(4.50, 2.45, 0.0, 1.55, 7.00), FuelClasses(750.0, 109.0, 30.0, 1800.0, 1500.0), _H8, 4.4, 0.40),
    # Timber-Understory
    TU1=FuelModel(161, FuelClasses(0.20, 0.90, 1.50, 0.20, 0.90), FuelClasses(2000.0, 109.0, 30.0, 1800.0, 1600.0), _H8, 0.6, 0.20),
    TU2=FuelModel(162, FuelClasses(0.95, 1.80, 1.25, 0.0, 0.20), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1600.0), _H8, 1.0, 0.30),
    TU3=FuelModel(163, FuelClasses(1.10, 0.15, 0.25, 0.65, 1.10), FuelClasses(1800.0, 109.0, 30.0, 1600.0, 1400.0), _H8, 1.3, 0.30),
    TU4=FuelModel(164, FuelClasses(4.50, 0.0, 0.0, 0.0, 2.00), FuelClasses(2300.0, 109.0, 30.0, 1500.0, 2000.0), _H8, 0.5, 0.12),
    TU5=FuelModel(165, FuelClasses(4.00, 4.00, 3.00, 0.0, 3.00), FuelClasses(1500.0, 109.0, 30.0, 1500.0, 750.0), _H8, 1.0, 0.25),
    # Timber Litter
    TL1=FuelModel(181, FuelClasses(1.00, 2.20, 3.60, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.2, 0.30),
    TL2=FuelModel(182, FuelClasses(1.40, 2.30, 2.20, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.2, 0.25),
    TL3=FuelModel(183, FuelClasses(0.50, 2.20, 2.80, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.20),
    TL4=FuelModel(184, FuelClasses(0.50, 1.50, 4.20, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.4, 0.25),
    TL5=FuelModel(185, FuelClasses(1.15, 2.50, 4.40, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.6, 0.25),
    TL6=FuelModel(186, FuelClasses(2.40, 1.20, 1.20, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.25),
    TL7=FuelModel(187, FuelClasses(0.30, 1.40, 8.10, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.4, 0.25),
    TL8=FuelModel(188, FuelClasses(5.80, 1.40, 1.10, 0.0, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.3, 0.35),
    TL9=FuelModel(189, FuelClasses(6.65, 3.30, 4.15, 0.0, 0.0), FuelClasses(1800.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 0.6, 0.35),
    # Slash-Blowdown
    SB1=FuelModel(201, FuelClasses(1.50, 3.00, 11.00, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.0, 0.25),
    SB2=FuelModel(202, FuelClasses(4.50, 4.25, 4.00, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.0, 0.25),
    SB3=FuelModel(203, FuelClasses(5.50, 2.75, 3.00, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 1.2, 0.25),
    SB4=FuelModel(204, FuelClasses(5.25, 3.50, 5.25, 0.0, 0.0), FuelClasses(2000.0, 109.0, 30.0, 1500.0, 1500.0), _H8, 2.7, 0.25),
)

# LANDFIRE codes with no Rothermel parameters (Both set δ = 0, so spread_rate short-circuits to 0):
#  - nonburnable: no fuel present.
#  - unknown: cover type present but unmodeled.
const _ZERO5 = FuelClasses(0.0, 0.0, 0.0, 0.0, 0.0)
const _NAN5 = FuelClasses(NaN, NaN, NaN, NaN, NaN)
nonburnable(code) = FuelModel(code, _ZERO5, _ZERO5, _ZERO5, 0.0, 0.0)
unknown(code) = FuelModel(code, _NAN5, _NAN5, _NAN5, 0.0, 0.0)

const LANDFIRE_ADDITIONAL_CODES = (
    NO_DATA = unknown(-9999),
    URBAN = unknown(91),
    SNOW_ICE = nonburnable(92),
    AGRICULTURE = unknown(93),
    WATER = nonburnable(98),
    BARREN = nonburnable(99)
)

const ALL_CODES = merge(NFFL, SB40, LANDFIRE_ADDITIONAL_CODES)

# Find FuelModel from its code.
function lookup(i::Int)
    k = findfirst(x -> x.code == i, ALL_CODES)
    isnothing(k) ? error("No Rothermel.FuelModel has code: $i") : ALL_CODES[k]
end

end  # module Rothermel
