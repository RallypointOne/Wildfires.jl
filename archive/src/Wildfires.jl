module Wildfires

using Dates, Terse, Extents
import GeoFormatTypes as GFT
import GeoInterface as GI

@types Component > (
    FuelSystem,
    Landscape > (
        FuelMap{T, A <: AbstractMatrix{T}, S <: FuelSystem}(data::A, system::S),
        Terrain{T, A <: AbstractMatrix{T}}(elev::A, slope::A = derive_slope(elev), aspect::A = derive_aspect(elev))
    )
)

#------------------------------------------------------------------------------# autoshow
# Show as: Type{Params, ...}(field1=val1, field2=val2, ...)
function autoshow(io::IO, x::T, fields = fieldnames(T)) where {T}
    params = T.parameters
    name = T.name.name
    print(io, name)
    isempty(params) || print(io, '{', join(params, ", "), '}')
    isempty(fields) || print(io, '(', ["$f=$(getfield(x, f))" for f in fields]..., ')')
end

#------------------------------------------------------------------------------# Grid
struct Grid{T, C <: GFT.CoordinateReferenceSystemFormat}
    x::Tuple{T, T}
    y::Tuple{T, T}
    dx::T
    dy::T
    crs::C
    start_utc::DateTime
    stop_utc::DateTime
end

end # module Wildfires
