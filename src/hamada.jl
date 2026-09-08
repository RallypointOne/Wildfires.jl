#-----------------------------------------------------------------------------# HamadaModel
"""
    HamadaModel(FT = Float64)

Coefficients of the Hamada (1951) urban fire spread model as used in HAZUS
and ELMFIRE: for each of the downwind, crosswind, and upwind directions, the
wind polynomial `c₁ (1 + c₂ V + c₃ V²)` and the separation-crossing term
`c₄ + c₅ V`, with `V` the open wind in m/s. `t_converge` is the time in
minutes after which the model's time-dependent extent is treated as a
steady rate.

Defaults are ELMFIRE's. Construct with keyword overrides to calibrate.

### Examples
```julia
HamadaModel()
HamadaModel(Float32)
HamadaModel(; c5 = (2.0, 0.25, 0.2))
```
"""
struct HamadaModel{T}
    c1::NTuple{3, T}
    c2::NTuple{3, T}
    c3::NTuple{3, T}
    c4::NTuple{3, T}
    c5::NTuple{3, T}
    t_converge::T
end

function HamadaModel(FT::Type{<:AbstractFloat} = Float64;
                     c1 = (1.6, 1.0, 1.0),
                     c2 = (0.1, 0.0, 0.0),
                     c3 = (0.007, 0.005, 0.002),
                     c4 = (25.0, 5.0, 5.0),
                     c5 = (2.5, 0.25, 0.2),
                     t_converge = 120.0)
    T3(x) = NTuple{3, FT}(x)
    return HamadaModel{FT}(T3(c1), T3(c2), T3(c3), T3(c4), T3(c5), FT(t_converge))
end
HamadaModel{T}(h::HamadaModel) where {T} =
    HamadaModel{T}(T.(h.c1), T.(h.c2), T.(h.c3), T.(h.c4), T.(h.c5), T(h.t_converge))

#-----------------------------------------------------------------------------# hamada_rates
"""
    hamada_rates(model::HamadaModel, wind, plan_dimension, separation, nonburnable_fraction)
        -> (downwind, crosswind, upwind)

Rates of spread [m/s] through a built area from the Hamada model: `wind` is
the open wind [m/s], `plan_dimension` the mean building plan dimension [m],
`separation` the mean building separation [m], and `nonburnable_fraction`
the fraction of fire-resistant buildings.

Ported from ELMFIRE, including the HAZUS blending toward isotropic spread
below 10 m/s and the fallback to the time-averaged extent when the model's
spread distance collapses. All three rates are positive and finite for any
non-negative inputs. The front advances one block pitch
`plan_dimension + separation` per building burn time, so wider separation
raises the rate; the fire-resistant fraction is what lowers it.

### Examples
```julia
hamada_rates(HamadaModel(), 15.0, 15.0, 10.0, 0.0)   # ≈ (0.27, 0.05, 0.03)
```
"""
function hamada_rates(h::HamadaModel{T}, wind, plan_dimension, separation, nonburnable_fraction) where {T}
    V = max(T(wind), zero(T))
    A = max(T(plan_dimension), zero(T))
    D = max(T(separation), zero(T))
    F = clamp(T(nonburnable_fraction), zero(T), one(T))
    X = h.t_converge
    ε = T(1e-10)

    # Minutes for a fully developed fire to reach the next building, per direction.
    τ = ntuple(3) do k
        cv = h.c1[k] * (1 + h.c2[k] * V + h.c3[k] * V^2)
        cross = _pos(h.c4[k] + h.c5[k] * V)
        ((1 - F) * (3 + T(0.375) * A + 8 * D / cross) +
              F * (5 + T(0.625) * A + 16 * D / cross)) / _pos(cv)
    end
    τ_d, τ_s, τ_u = map(_pos, τ)

    # Spread distance after X minutes, and the steady rate, in metres and m/min.
    K_d = max((A + D) / τ_d * X, ε)
    K_s = max((A / 2 + D) + (A + D) / τ_s * (X - τ_s), ε)
    K_u = max((A / 2 + D) + (A + D) / τ_u * (X - τ_u), ε)
    V_d = max((A + D) / τ_d, ε)
    V_s = max((A + D) / τ_s, ε)
    V_u = max((A + D) / τ_u, ε)

    # HAZUS: below 10 m/s blend each direction toward a common geometric mean.
    w = clamp(V / 10, zero(T), one(T))
    blend = (K_d * V_s + V_d * K_s + K_u * V_s + V_u * K_s) *
            sqrt(2 / _pos(K_d + K_u) / _pos(K_s)) * (1 - w) / 4
    V_d = max(V_d * w + blend, ε)
    V_s = max(V_s * w + blend, ε)
    V_u = max(V_u * w + blend, ε)

    # ELMFIRE's fallback when the spread distance collapses.
    small = min(K_d, K_s, K_u) <= T(0.1)
    V_d = ifelse(small, K_d / _pos(X), V_d)
    V_s = ifelse(small, K_s / _pos(X), V_s)
    V_u = ifelse(small, K_u / _pos(X), V_u)

    return (V_d / 60, V_s / 60, V_u / 60)
end

#-----------------------------------------------------------------------------# ellipse_speed
"""
    ellipse_speed(nx, ny, wx, wy, head, flank, back)

Normal speed of a front whose Huygens wavelet is the ellipse with semi-axis
`(head + back) / 2` along the unit wind direction `(wx, wy)`, semi-axis
`flank` across it, and centre offset `(head - back) / 2` downwind: the
support function of that ellipse in the direction of the unit normal
`(nx, ny)`. Equals `head` downwind, `back` upwind, and `flank` crosswind.

### Examples
```julia
ellipse_speed(1.0, 0.0, 1.0, 0.0, 2.0, 0.5, 0.2)   # 2.0
ellipse_speed(0.0, 1.0, 1.0, 0.0, 2.0, 0.5, 0.2)   # 0.5
```
"""
@inline function ellipse_speed(nx, ny, wx, wy, head, flank, back)
    a = (head + back) / 2
    c = (head - back) / 2
    cosθ = nx * wx + ny * wy
    sin²θ = max(1 - cosθ^2, zero(cosθ))
    return c * cosθ + sqrt(a^2 * cosθ^2 + flank^2 * sin²θ)
end
