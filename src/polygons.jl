import GeoInterface as GI
using Oceananigans.Grids: xspacings, yspacings

#-----------------------------------------------------------------------------# Geometry helpers
# Iterate `(item, exterior ring)` pairs over polygons, features, and feature
# collections; multipolygons yield one pair per polygon.
function _rings(geometries)
    items = GI.trait(geometries) isa GI.FeatureCollectionTrait ? GI.getfeature(geometries) : geometries
    return Iterators.flatten(_rings_of(item) for item in items)
end

function _rings_of(item)
    geom = GI.trait(item) isa GI.FeatureTrait ? GI.geometry(item) : item
    trait = GI.trait(geom)
    if trait isa GI.PolygonTrait
        return ((item, GI.getexterior(geom)),)
    elseif trait isa GI.MultiPolygonTrait
        return ((item, GI.getexterior(poly)) for poly in GI.getgeom(geom))
    else
        return ()
    end
end

# Exterior ring as a closed vector of (x, y) tuples in grid metres.
function _closed_ring(ring, transform)
    pts = [transform(GI.x(p), GI.y(p)) for p in GI.getpoint(ring)]
    first(pts) == last(pts) || push!(pts, first(pts))
    return pts
end

# Shoelace area (absolute) and centroid of a closed ring of (x, y) tuples.
function _area_centroid(pts)
    A2 = cx = cy = 0.0
    for k in 1:length(pts)-1
        (x₁, y₁), (x₂, y₂) = pts[k], pts[k+1]
        w = x₁ * y₂ - x₂ * y₁
        A2 += w
        cx += (x₁ + x₂) * w
        cy += (y₁ + y₂) * w
    end
    A2 == 0 && return (0.0, 0.0, 0.0)
    return (abs(A2) / 2, cx / (3A2), cy / (3A2))
end

# Index of the cell containing coordinate `q`, given the first face and spacing.
_cell(q, q₀, Δ) = floor(Int, (q - q₀) / Δ) + 1

# Even-odd ray casting against a closed ring.
function _inside(pts, x, y)
    inside = false
    for k in 1:length(pts)-1
        (x₁, y₁), (x₂, y₂) = pts[k], pts[k+1]
        if (y₁ > y) != (y₂ > y)
            xcross = x₁ + (y - y₁) / (y₂ - y₁) * (x₂ - x₁)
            inside ⊻= x < xcross
        end
    end
    return inside
end

# Distance from (x, y) to the nearest edge of a closed ring.
function _edge_distance(pts, x, y)
    d² = Inf
    for k in 1:length(pts)-1
        (x₁, y₁), (x₂, y₂) = pts[k], pts[k+1]
        ex, ey = x₂ - x₁, y₂ - y₁
        s = clamp(((x - x₁) * ex + (y - y₁) * ey) / max(ex^2 + ey^2, 1e-300), 0.0, 1.0)
        d² = min(d², (x - x₁ - s * ex)^2 + (y - y₁ - s * ey)^2)
    end
    return sqrt(d²)
end

# Fraction of the cell with lower-left corner (x, y) covered by the ring, on
# an n × n lattice of sample points.
function _coverage(pts, x, y, Δx, Δy, n)
    hits = 0
    for q in 1:n, p in 1:n
        hits += _inside(pts, x + (p - 0.5) * Δx / n, y + (q - 0.5) * Δy / n)
    end
    return hits / n^2
end

# Horizontal geometry of a fire grid: sizes, first faces, spacings.
function _grid_geometry(grid)
    topology(grid)[3] === Flat ||
        throw(ArgumentError("expected a horizontal grid with Flat vertical topology; got $(topology(grid))"))
    xf, yf = xnodes(grid, Face()), ynodes(grid, Face())
    Δx, Δy = first(xspacings(grid, Center())), first(yspacings(grid, Center()))
    return (Nx = size(grid, 1), Ny = size(grid, 2), xf, yf, Δx, Δy)
end

# Rasterize polygons onto the grid. Returns the per-cell coverage (summed over
# polygons) and, per polygon that touches the grid, its item, centroid, area,
# and covered cells in compressed-row form.
function _footprints(geometries, crs, grid; source_crs, subsamples)
    (; Nx, Ny, xf, yf, Δx, Δy) = _grid_geometry(grid)
    transform = to_grid_transform(crs, source_crs)
    x₀, y₀ = first(xf), first(yf)

    items = Any[]
    xs, ys, areas = Float64[], Float64[], Float64[]
    cell_i, cell_j = Int32[], Int32[]
    offsets = Int32[1]
    coverage = zeros(Float64, Nx, Ny)

    for (item, ring) in _rings(geometries)
        pts = _closed_ring(ring, transform)
        A, cx, cy = _area_centroid(pts)
        A < 1 && continue

        xlo, xhi = extrema(first, pts)
        ylo, yhi = extrema(last, pts)
        ilo, ihi = _cell(xlo, x₀, Δx), _cell(xhi, x₀, Δx)
        jlo, jhi = _cell(ylo, y₀, Δy), _cell(yhi, y₀, Δy)
        (ihi < 1 || ilo > Nx || jhi < 1 || jlo > Ny) && continue

        ncells = 0
        for j in max(jlo, 1):min(jhi, Ny), i in max(ilo, 1):min(ihi, Nx)
            f = _coverage(pts, xf[i], yf[j], Δx, Δy, subsamples)
            f > 0 || continue
            coverage[i, j] += f
            push!(cell_i, i)
            push!(cell_j, j)
            ncells += 1
        end
        ncells > 0 || continue

        push!(items, item)
        push!(xs, cx); push!(ys, cy); push!(areas, A)
        push!(offsets, last(offsets) + ncells)
    end

    return (; items, xs, ys, areas, cell_i, cell_j, offsets, coverage)
end

# A horizontal field holding a CPU array, converted to the grid's float type.
function _horizontal_field(grid, A)
    f = Field{Center, Center, Nothing}(grid)
    set!(f, reshape(eltype(grid).(A), size(grid, 1), size(grid, 2), 1))
    return f
end

#-----------------------------------------------------------------------------# polygon_coverage
"""
    polygon_coverage(geometries, crs::ProjectedCRS, grid;
                     source_crs = "EPSG:4326", subsamples = 4) -> Field

Fraction of each cell of the horizontal `grid` covered by the polygons in
`geometries` (any GeoInterface-compatible polygons, features, or feature
collection in `source_crs`), sampled on a `subsamples × subsamples` lattice
per cell using exterior rings only.

### Examples
```julia
observed = polygon_coverage(GeoJSON.read("perimeter.geojson"), crs, model.grid)
burned_inside = count(isfinite.(interior(model.t_ignition)) .& (interior(observed) .> 0.5))
```
"""
function polygon_coverage(geometries, crs::ProjectedCRS, grid;
                          source_crs = "EPSG:4326", subsamples::Integer = 4)
    (; coverage) = _footprints(geometries, crs, grid; source_crs, subsamples)
    return _horizontal_field(grid, min.(coverage, 1))
end

#-----------------------------------------------------------------------------# ignite! from polygons
"""
    ignite!(model, geometries, crs::ProjectedCRS; time = model.clock.time, source_crs = "EPSG:4326")

Ignite the interior of the polygons in `geometries` (any GeoInterface-compatible
polygons, features, or feature collection in `source_crs`), for instance an
observed fire perimeter.

`φ` is set to the pointwise minimum of its current value and the exact signed
distance to the polygons' exterior rings — negative inside — so an observed
perimeter can also re-initialize a running fire. Cells inside get
`t_ignition = time` unless already ignited earlier. The signed distance is
computed on the CPU for every cell, which costs cells × edges.

### Examples
```julia
ignite!(model, GeoJSON.read("perimeter.geojson"), crs; time = 6 * 3600)
```
"""
function ignite!(model::FireModel, geometries, crs::ProjectedCRS;
                 time = model.clock.time, source_crs = "EPSG:4326")
    grid = model.grid
    (; Nx, Ny) = _grid_geometry(grid)
    transform = to_grid_transform(crs, source_crs)
    rings = [_closed_ring(ring, transform) for (_, ring) in _rings(geometries)]
    xc, yc = xnodes(grid, Center()), ynodes(grid, Center())

    d = fill(Inf, Nx, Ny)
    Threads.@threads for j in 1:Ny
        for i in 1:Nx, pts in rings
            δ = _edge_distance(pts, xc[i], yc[j])
            d[i, j] = min(d[i, j], ifelse(_inside(pts, xc[i], yc[j]), -δ, δ))
        end
    end

    FT = eltype(grid)
    launch!(architecture(grid), grid, :xy, _ignite_distance!, model.φ, model.t_ignition,
            _horizontal_field(grid, d), FT(time))
    fill_halo_regions!(model.φ)
    fill_halo_regions!(model.t_ignition)
    return model
end

@kernel function _ignite_distance!(φ, t_ignition, d, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        dᵢ = d[i, j, 1]
        φ[i, j, 1] = min(φ[i, j, 1], dᵢ)
        t_ignition[i, j, 1] = ifelse(dᵢ < 0, min(t_ignition[i, j, 1], time), t_ignition[i, j, 1])
    end
end
