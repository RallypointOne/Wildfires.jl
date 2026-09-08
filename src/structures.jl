import GeoInterface as GI
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: xspacings, yspacings

#-----------------------------------------------------------------------------# BuildingClass
"""
    BuildingClass(; fire_resistant = false, peak_hrr = 25e3,
                    t_growth = 400, t_plateau = 10080, t_decay = 14400)

Properties shared by a class of structures: whether the class counts as
fire-resistant in the Hamada spread model, and the heat-release curve of a
burning structure per square metre of footprint — `peak_hrr` [W/m²] reached
`t_growth` seconds after ignition, held until `t_plateau`, then falling
linearly to zero at `t_decay`. Defaults are ELMFIRE's building fuel model 1.

### Examples
```julia
BuildingClass()
BuildingClass(fire_resistant = true, peak_hrr = 25e3)
```
"""
struct BuildingClass{T}
    fire_resistant::Bool
    peak_hrr::T
    t_growth::T
    t_plateau::T
    t_decay::T
end

function BuildingClass(; fire_resistant = false, peak_hrr = 25e3,
                         t_growth = 400.0, t_plateau = 10080.0, t_decay = 14400.0)
    0 < t_growth <= t_plateau <= t_decay ||
        throw(ArgumentError("need 0 < t_growth ≤ t_plateau ≤ t_decay"))
    return BuildingClass(fire_resistant, promote(float(peak_hrr), t_growth, t_plateau, t_decay)...)
end
BuildingClass{T}(c::BuildingClass) where {T} =
    BuildingClass(c.fire_resistant, T(c.peak_hrr), T(c.t_growth), T(c.t_plateau), T(c.t_decay))

# Heat release rate per unit footprint area `τ` seconds after ignition.
@inline function _hrr(c::BuildingClass{T}, τ) where {T}
    z = zero(T)
    growth = c.peak_hrr * τ / c.t_growth
    decay = c.peak_hrr * (τ - c.t_decay) / (c.t_plateau - c.t_decay)
    q = ifelse(τ < c.t_growth, growth, ifelse(τ <= c.t_plateau, c.peak_hrr, decay))
    return ifelse((τ < z) | (τ > c.t_decay), z, q)
end

#-----------------------------------------------------------------------------# Structures
"""
    Structures(geometries, crs::ProjectedCRS, grid;
               source_crs = "EPSG:4326", classes = (BuildingClass(),),
               class = Returns(1), radius = 60, subsamples = 4)

Building footprints as first-class objects on the fire `grid`.

`geometries` is any GeoInterface-compatible collection of polygons or
features (a GeoJSON `FeatureCollection`, a vector of polygons, ...) in
`source_crs`; each polygon of a multipolygon becomes its own structure and
polygons that do not touch the grid are dropped. `class(item)` returns the
index into `classes` for each item, so attributes such as construction type
can be mapped to a [`BuildingClass`](@ref).

Per structure, on the grid's architecture: centroid `x`, `y` [grid m],
footprint `area` [m²], `class`, and `t_ignition` (`Inf` until the fire front
reaches any of its footprint cells; see [`update_structures!`](@ref)).
`cell_i`, `cell_j`, `offsets` list each structure's footprint cells in
compressed-row form.

Per fire-grid cell, as fields: `footprint_fraction`, the area fraction under
footprints, and three neighbourhood means over the structures within
`radius` metres of the cell centre — `plan_dimension` (√area) [m],
`separation` (edge-to-edge distance to the nearest other structure, capped
at `radius`) [m], and `nonburnable_fraction` (share of fire-resistant
classes). These are the inputs of [`hamada_rates`](@ref); cells with no
structure within `radius` hold zero and are not urban.

Footprint coverage is sampled on a `subsamples × subsamples` lattice per
cell using the exterior ring only. Construction runs on the CPU.

### Examples
```julia
fc = GeoJSON.read("buildings.geojson")
structures = Structures(fc, crs, model.grid;
                        class = f -> f.building == "industrial" ? 2 : 1,
                        classes = (BuildingClass(), BuildingClass(fire_resistant = true)))
```
"""
struct Structures{G, C, VF, VI, F}
    grid::G
    classes::C
    x::VF
    y::VF
    area::VF
    class::VI
    t_ignition::VF
    cell_i::VI
    cell_j::VI
    offsets::VI
    footprint_fraction::F
    plan_dimension::F
    separation::F
    nonburnable_fraction::F
end

Base.length(s::Structures) = length(s.area)

function Base.show(io::IO, s::Structures)
    print(io, "Structures: ", length(s), " on ", summary(s.grid), "\n",
              "├── classes: ", length(s.classes), "\n",
              "└── fields: footprint_fraction, plan_dimension, separation, nonburnable_fraction")
end

function Structures(geometries, crs::ProjectedCRS, grid;
                    source_crs = "EPSG:4326", classes = (BuildingClass(),),
                    class = Returns(1), radius = 60.0, subsamples::Integer = 4)
    topology(grid)[3] === Flat ||
        throw(ArgumentError("Structures requires a horizontal grid with Flat vertical topology"))
    FT = eltype(grid)
    classes = Tuple(BuildingClass{FT}(c) for c in classes)
    transform = to_grid_transform(crs, source_crs)

    Nx, Ny = size(grid, 1), size(grid, 2)
    xf, yf = xnodes(grid, Face()), ynodes(grid, Face())
    Δx, Δy = first(xspacings(grid, Center())), first(yspacings(grid, Center()))
    x₀, y₀ = first(xf), first(yf)

    xs, ys, areas = Float64[], Float64[], Float64[]
    cls = Int32[]
    cell_i, cell_j = Int32[], Int32[]
    offsets = Int32[1]
    coverage = zeros(Float64, Nx, Ny)

    for (item, ring) in _rings(geometries)
        pts = [transform(GI.x(p), GI.y(p)) for p in GI.getpoint(ring)]
        first(pts) == last(pts) || push!(pts, first(pts))
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

        push!(xs, cx); push!(ys, cy); push!(areas, A)
        push!(cls, class(item))
        push!(offsets, last(offsets) + ncells)
    end

    n = length(areas)
    all(1 <= c <= length(classes) for c in cls) ||
        throw(ArgumentError("class index out of range for $(length(classes)) classes"))
    plan = sqrt.(areas)
    sep = _separations(xs, ys, plan, radius)
    resistant = [classes[c].fire_resistant for c in cls]

    # Neighbourhood means over structures within `radius` of each cell centre.
    xc, yc = xnodes(grid, Center()), ynodes(grid, Center())
    Σplan, Σsep, Σres, count = (zeros(Float64, Nx, Ny) for _ in 1:4)
    for s in 1:n
        for j in max(_cell(ys[s] - radius, y₀, Δy), 1):min(_cell(ys[s] + radius, y₀, Δy), Ny),
            i in max(_cell(xs[s] - radius, x₀, Δx), 1):min(_cell(xs[s] + radius, x₀, Δx), Nx)
            (xc[i] - xs[s])^2 + (yc[j] - ys[s])^2 <= radius^2 || continue
            Σplan[i, j] += plan[s]
            Σsep[i, j] += sep[s]
            Σres[i, j] += resistant[s]
            count[i, j] += 1
        end
    end
    mean(Σ) = ifelse.(count .> 0, Σ ./ max.(count, 1), 0.0)

    field(A) = (f = Field{Center, Center, Nothing}(grid); set!(f, reshape(FT.(A), Nx, Ny, 1)); f)
    arch = architecture(grid)
    dev(v) = on_architecture(arch, v)

    return Structures(grid, classes,
                      dev(FT.(xs)), dev(FT.(ys)), dev(FT.(areas)), dev(cls),
                      dev(fill(FT(Inf), n)), dev(cell_i), dev(cell_j), dev(offsets),
                      field(min.(coverage, 1)), field(mean(Σplan)), field(mean(Σsep)), field(mean(Σres)))
end

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

# Fraction of the cell with lower-left corner (x, y) covered by the ring, on
# an n × n lattice of sample points.
function _coverage(pts, x, y, Δx, Δy, n)
    hits = 0
    for q in 1:n, p in 1:n
        hits += _inside(pts, x + (p - 0.5) * Δx / n, y + (q - 0.5) * Δy / n)
    end
    return hits / n^2
end

# Edge-to-edge distance from each structure to its nearest neighbour within
# `radius`, estimated from centroid distance and plan dimensions; `radius`
# when there is none. Bins of size `radius` keep the search local.
function _separations(xs, ys, plan, radius)
    n = length(xs)
    sep = fill(Float64(radius), n)
    n == 0 && return sep
    x₀, y₀ = minimum(xs), minimum(ys)
    bin(x, y) = (floor(Int, (x - x₀) / radius), floor(Int, (y - y₀) / radius))
    bins = Dict{NTuple{2, Int}, Vector{Int}}()
    for s in 1:n
        push!(get!(bins, bin(xs[s], ys[s]), Int[]), s)
    end
    for s in 1:n
        bi, bj = bin(xs[s], ys[s])
        d² = radius^2
        nearest = 0
        for dj in -1:1, di in -1:1
            for t in get(bins, (bi + di, bj + dj), Int[])
                t == s && continue
                r² = (xs[t] - xs[s])^2 + (ys[t] - ys[s])^2
                r² < d² && (d² = r²; nearest = t)
            end
        end
        nearest == 0 && continue
        sep[s] = clamp(sqrt(d²) - (plan[s] + plan[nearest]) / 2, 0.0, radius)
    end
    return sep
end

#-----------------------------------------------------------------------------# update_structures!
"""
    update_structures!(structures::Structures, model::FireModel)

Set each structure's `t_ignition` to the earliest ignition time among its
footprint cells in `model`, leaving it `Inf` while the front has reached
none of them. Call after [`advance!`](@ref).

### Examples
```julia
advance!(model, Δt; speed)
update_structures!(structures, model)
count(isfinite, Array(structures.t_ignition))   # structures reached so far
```
"""
function update_structures!(structures::Structures, model::FireModel)
    n = length(structures)
    n == 0 && return structures
    grid = structures.grid
    launch!(architecture(grid), grid, (n,), _structure_ignition!,
            structures.t_ignition, structures.offsets, structures.cell_i, structures.cell_j,
            model.t_ignition)
    return structures
end

@kernel function _structure_ignition!(t_structure, offsets, cell_i, cell_j, t_ignition)
    s = @index(Global)
    @inbounds begin
        t = t_structure[s]
        for k in offsets[s]:offsets[s+1]-1
            t = min(t, t_ignition[cell_i[k], cell_j[k], 1])
        end
        t_structure[s] = t
    end
end

#-----------------------------------------------------------------------------# heat_release
"""
    heat_release!(hrr, structures::Structures, t)
    heat_release(structures::Structures, t)

Heat release rate [W] of every structure at time `t`: footprint area times
its class's curve evaluated `t - t_ignition` seconds after ignition, zero
for structures not yet ignited or burnt out. The allocating form returns a
new vector on the grid's architecture.

### Examples
```julia
sum(heat_release(structures, model.clock.time))   # total W
```
"""
function heat_release!(hrr, structures::Structures, t)
    n = length(structures)
    n == 0 && return hrr
    grid = structures.grid
    launch!(architecture(grid), grid, (n,), _heat_release!, hrr, structures.area,
            structures.class, structures.t_ignition, structures.classes, eltype(grid)(t))
    return hrr
end

heat_release(structures::Structures, t) = heat_release!(similar(structures.area), structures, t)

@kernel function _heat_release!(hrr, area, class, t_ignition, classes, t)
    s = @index(Global)
    @inbounds begin
        ignited = isfinite(t_ignition[s])
        τ = ifelse(ignited, t - t_ignition[s], zero(t))
        hrr[s] = ifelse(ignited, area[s] * _hrr(classes[class[s]], τ), zero(t))
    end
end
