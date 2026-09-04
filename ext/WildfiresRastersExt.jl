module WildfiresRastersExt

using Wildfires
using Rasters
using GeoFormatTypes: EPSG

#-----------------------------------------------------------------------------# Bilinear sampling
# Fractional index of `q` in a regularly spaced lookup starting at `q₁` with
# spacing `Δ`. Works for a descending lookup (north-up rasters have Δ < 0).
@inline _findex(q, q₁, Δ) = (q - q₁) / Δ + 1

# Bilinear interpolation of `A` at fractional indices (fi, fj), or `fill` when
# the sample falls outside the array.
@inline function _bilinear(A, fi, fj, fill)
    ni, nj = size(A)
    (fi < 1 || fi > ni || fj < 1 || fj > nj) && return fill
    i = clamp(floor(Int, fi), 1, ni - 1)
    j = clamp(floor(Int, fj), 1, nj - 1)
    s = fi - i
    t = fj - j
    @inbounds return (1 - s) * (1 - t) * A[i, j] + s * (1 - t) * A[i+1, j] +
                     (1 - s) * t * A[i, j+1] + s * t * A[i+1, j+1]
end

#-----------------------------------------------------------------------------# raster_topography
# Reprojecting up front keeps Proj out of the returned closure: `set!` evaluates
# it from multiple threads, and a `Proj.Transformation` is not thread-safe.
function Wildfires.raster_topography(raster::AbstractRaster, crs::ProjectedCRS;
                                     fill_value = 0)
    projected = resample(raster; crs = EPSG(crs.epsg), method = :bilinear)
    A = parent(read(replace_missing(projected, fill_value)))
    Xs = lookup(projected, X)
    Ys = lookup(projected, Y)
    x₁, Δx = first(Xs), step(Xs)
    y₁, Δy = first(Ys), step(Ys)
    x₀, y₀ = crs.origin
    return (x, y) -> _bilinear(A, _findex(x + x₀, x₁, Δx), _findex(y + y₀, y₁, Δy), fill_value)
end

end # module WildfiresRastersExt
