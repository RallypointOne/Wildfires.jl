module WildfiresRastersExt

using Wildfires
using Rasters
using Rasters.DimensionalData.Lookups: locus, Start, End
using GeoFormatTypes: EPSG

#-----------------------------------------------------------------------------# Sampling
# Fractional index of `q` in a regularly spaced lookup whose first cell centre
# is `q₁` with spacing `Δ`. Works for a descending lookup (north-up rasters
# have Δ < 0). Integer values fall on cell centres.
@inline _findex(q, q₁, Δ) = (q - q₁) / Δ + 1

# GDAL rasters carry the cell *start* as the lookup value; shift to the centre.
_centre_offset(l) = locus(l) isa Start ? 0.5 : locus(l) isa End ? -0.5 : 0.0

# Inside the raster means within half a cell of the outermost centres.
@inline _inside(fi, fj, ni, nj) = 0.5 <= fi < ni + 0.5 && 0.5 <= fj < nj + 0.5

@inline function _nearest(A, fi, fj, fill)
    ni, nj = size(A)
    _inside(fi, fj, ni, nj) || return fill
    @inbounds return A[round(Int, fi), round(Int, fj)]
end

# Bilinear between the four surrounding centres; the outer half-cell strip
# clamps to the edge centres.
@inline function _bilinear(A, fi, fj, fill)
    ni, nj = size(A)
    _inside(fi, fj, ni, nj) || return fill
    fi = clamp(fi, 1, ni)
    fj = clamp(fj, 1, nj)
    i = clamp(floor(Int, fi), 1, ni - 1)
    j = clamp(floor(Int, fj), 1, nj - 1)
    s = fi - i
    t = fj - j
    @inbounds return (1 - s) * (1 - t) * A[i, j] + s * (1 - t) * A[i+1, j] +
                     (1 - s) * t * A[i, j+1] + s * t * A[i+1, j+1]
end

_interpolant(::Val{:near}) = _nearest
_interpolant(::Val{:bilinear}) = _bilinear

#-----------------------------------------------------------------------------# raster_sampler
# Reprojecting up front keeps Proj out of the returned closure: `set!` evaluates
# it from multiple threads, and a `Proj.Transformation` is not thread-safe.
function Wildfires.raster_sampler(raster::AbstractRaster, crs::ProjectedCRS;
                                  method = :bilinear, fill_value = 0)
    method in (:near, :bilinear) ||
        throw(ArgumentError("method must be :near or :bilinear, got $method"))
    interpolant = _interpolant(Val(method))

    projected = resample(raster; crs = EPSG(crs.epsg), method)
    A = parent(read(replace_missing(projected, fill_value)))
    Xs = lookup(projected, X)
    Ys = lookup(projected, Y)
    Δx, Δy = step(Xs), step(Ys)
    x₁ = first(Xs) + _centre_offset(Xs) * Δx
    y₁ = first(Ys) + _centre_offset(Ys) * Δy
    x₀, y₀ = crs.origin
    return (x, y) -> interpolant(A, _findex(x + x₀, x₁, Δx), _findex(y + y₀, y₁, Δy), fill_value)
end

end # module WildfiresRastersExt
