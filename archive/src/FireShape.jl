module FireShape

export AbstractFireShape, CosineShape, EllipticalShape
export length_to_breadth, fire_eccentricity
export normal_speed, radial_speed

#-----------------------------------------------------------------------------# AbstractFireShape
"""
    AbstractFireShape

Supertype for directional fire spread shapes that control how the spread rate
varies with angle relative to the dominant push direction (wind + slope).

Subtypes must implement:
- `normal_speed(shape, R_head, R_base, cos_theta, LB)` — for level set propagation
- `radial_speed(shape, R_head, R_base, cos_theta, LB)` — for CA propagation (Huygens wavelet)
"""
abstract type AbstractFireShape end

#-----------------------------------------------------------------------------# CosineShape
"""
    CosineShape()

Cosine-based directional spread model.

    R(θ) = R_base + (R_head - R_base) · max(0, cos θ)
"""
struct CosineShape <: AbstractFireShape end

#-----------------------------------------------------------------------------# EllipticalShape
"""
    EllipticalShape(; formula=:anderson)

Elliptical fire spread model. The fire perimeter is approximated as an ellipse
with the length-to-breadth ratio `LB` computed from effective wind speed.

Supported `formula` values for [`length_to_breadth`](@ref):
- `:anderson` — Anderson (1983), default
- `:green` — Green (1983)
"""
struct EllipticalShape <: AbstractFireShape
    formula::Symbol
end
EllipticalShape(; formula::Symbol = :anderson) = EllipticalShape(formula)

#-----------------------------------------------------------------------------# length_to_breadth
"""
    length_to_breadth(U; formula=:anderson)

Compute the fire length-to-breadth ratio from effective midflame wind speed `U` [m/s].

### Examples
```julia
length_to_breadth(5.0)                  # Anderson formula
length_to_breadth(5.0; formula=:green)  # Green formula
```
"""
function length_to_breadth(U; formula::Symbol = :anderson)
    if formula === :anderson
        LB = oftype(U, 0.936) * exp(oftype(U, 0.2566) * U) +
             oftype(U, 0.461) * exp(oftype(U, -0.1548) * U) -
             oftype(U, 0.397)
    elseif formula === :green
        LB = oftype(U, 1.1) * U^oftype(U, 0.464)
    else
        error("Unknown LB formula: $formula. Use :anderson or :green.")
    end
    return max(LB, one(LB))
end

#-----------------------------------------------------------------------------# fire_eccentricity
"""
    fire_eccentricity(LB)

Compute fire eccentricity from the length-to-breadth ratio `LB`.

### Examples
```julia
fire_eccentricity(2.0)  # ≈ 0.866
```
"""
fire_eccentricity(LB) = sqrt(LB^2 - one(LB)) / LB

#-----------------------------------------------------------------------------# normal_speed (level set)
"""
    normal_speed(shape, R_head, R_base, cos_theta, LB)

Compute the normal speed ``F_n(\\theta)`` for the level set equation
``\\partial\\phi/\\partial t + F_n|\\nabla\\phi| = 0``.

- `R_head` — head-fire rate of spread [m/min]
- `R_base` — backing rate of spread [m/min]
- `cos_theta` — cosine of angle between gradient normal and push direction
- `LB` — length-to-breadth ratio (used by [`EllipticalShape`](@ref), ignored by [`CosineShape`](@ref))
"""
function normal_speed(::CosineShape, R_head, R_base, cos_theta, LB)
    R_head == 0 && return zero(R_head)
    R_head ≈ R_base && return R_head
    return R_base + (R_head - R_base) * max(zero(cos_theta), cos_theta)
end

function normal_speed(::EllipticalShape, R_head, R_base, cos_theta, LB)
    R_head == 0 && return zero(R_head)
    R_head ≈ R_base && return R_head
    ε = fire_eccentricity(LB)
    sin2 = one(cos_theta) - cos_theta^2
    R_expand = R_head / (one(ε) + ε)
    F_n = R_expand * (sqrt(cos_theta^2 + sin2 / LB^2) + ε * cos_theta)
    return max(F_n, R_base)
end

#-----------------------------------------------------------------------------# radial_speed (CA / Huygens wavelet)
"""
    radial_speed(shape, R_head, R_base, cos_theta, LB)

Compute the radial speed ``v_r(\\alpha)`` for the CA Huygens wavelet approach.

Unlike [`normal_speed`](@ref), the radial speed gives the distance-per-time along the
ray from the fire source to each neighbor. Using `normal_speed` as `radial_speed`
overestimates flanking spread.

- `R_head` — head-fire rate of spread [m/min]
- `R_base` — backing rate of spread [m/min]
- `cos_theta` — cosine of angle between neighbor direction and push direction
- `LB` — length-to-breadth ratio (used by [`EllipticalShape`](@ref), ignored by [`CosineShape`](@ref))
"""
function radial_speed(::CosineShape, R_head, R_base, cos_theta, LB)
    R_head == 0 && return zero(R_head)
    R_head ≈ R_base && return R_head
    cos_theta <= zero(cos_theta) && return R_base
    sin_alpha = sqrt(max(zero(cos_theta), one(cos_theta) - cos_theta^2))
    return R_base / max(sin_alpha, R_base / R_head)
end

function radial_speed(::EllipticalShape, R_head, R_base, cos_theta, LB)
    R_head == 0 && return zero(R_head)
    R_head ≈ R_base && return R_head
    ε = fire_eccentricity(LB)
    v_r = R_head * (one(ε) - ε) / (one(ε) - ε * cos_theta)
    return max(v_r, R_base)
end

end # module
