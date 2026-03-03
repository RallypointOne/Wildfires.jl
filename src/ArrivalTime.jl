module ArrivalTime

export estimate_arrival_times, isochrones, perimeter_to_grid
export hausdorff_distance, jaccard_index, sorensen_dice, area_error

using ..LevelSet: LevelSetGrid, xcoords, ycoords, reinitialize!,
    AbstractReinitMethod, IterativeReinit
using ..SpreadModel: FireSpreadModel, directional_speed
using DataStructures: PriorityQueue

#--------------------------------------------------------------------------------# estimate_arrival_times
"""
    estimate_arrival_times(grid::LevelSetGrid, ignition_x, ignition_y, duration;
                           model=nothing, time_of_max_extent=duration,
                           normalize=model === nothing, passes=model isa FireSpreadModel ? 2 : 1)

Estimate fire arrival times via geodesic distance from the ignition point.

Given a `LevelSetGrid` where `φ < 0` defines the burned region (final perimeter),
computes the shortest-path (geodesic) distance from the ignition point to every
burned cell, constrained to stay within the burned region.

When `model` is a `FireSpreadModel`, edge weights use direction-dependent spread rates
via [`directional_speed`](@ref), producing physically realistic arrival times that
account for wind/slope direction.  A two-pass Dijkstra (default) refines the time
estimates by re-evaluating spread rates at the first-pass arrival times.

When `model` is a generic callable `model(t, x, y) -> spread_rate`, edge weights use
isotropic spread rates at the edge midpoint.

# Arguments
- `grid` — A `LevelSetGrid` where `φ < 0` defines the burned region
- `ignition_x, ignition_y` — Ignition point coordinates [m]
- `duration` — Total burn duration [min] (used for normalizing arrival times)
- `model` — Optional spread model for weighted distances
- `time_of_max_extent` — Time at which the final perimeter was observed [min]
- `normalize` — If `true`, linearly scale to `[0, time_of_max_extent]`. Default: `true`
  when `model === nothing`, `false` when a model is provided (raw costs are physical times).
- `passes` — Number of Dijkstra passes for iterative time refinement. Default: `2` for
  `FireSpreadModel`, `1` otherwise.

# Returns
A `Matrix{Float64}` of arrival times (same size as grid; `Inf` for unburned cells).

### Examples
```julia
grid = LevelSetGrid(100, 100, dx=30.0)
ignite!(grid, 1500.0, 1500.0, 100.0)
simulate!(grid, model, steps=200)
T = estimate_arrival_times(grid, 1500.0, 1500.0, 100.0)
```
"""
function estimate_arrival_times(grid::LevelSetGrid, ignition_x, ignition_y, duration;
                                model=nothing, time_of_max_extent=duration,
                                normalize::Bool = model === nothing,
                                passes::Int = model isa FireSpreadModel ? 2 : 1)
    ny, nx = size(grid)
    xs = collect(xcoords(grid))
    ys = collect(ycoords(grid))
    dx, dy = grid.dx, grid.dy

    # Find the burned cell nearest to the ignition point
    burned_mask = grid.φ .< 0
    best_i, best_j = 0, 0
    best_dist = Inf
    for j in 1:nx, i in 1:ny
        burned_mask[i, j] || continue
        d = hypot(xs[j] - ignition_x, ys[i] - ignition_y)
        if d < best_dist
            best_dist = d
            best_i, best_j = i, j
        end
    end
    best_i == 0 && error("No burned cells found in grid")

    t_mid = duration / 2

    # Multi-pass Dijkstra: each pass uses previous arrival times for time-dependent evaluation
    arrival = nothing
    for _ in 1:passes
        arrival = _dijkstra_pass(burned_mask, xs, ys, dx, dy, best_i, best_j,
                                 model, t_mid, arrival)
    end

    # Normalize to [0, time_of_max_extent] when requested
    if normalize
        max_cost = -Inf
        for j in 1:nx, i in 1:ny
            if isfinite(arrival[i, j])
                max_cost = max(max_cost, arrival[i, j])
            end
        end
        if max_cost > 0
            for j in 1:nx, i in 1:ny
                if isfinite(arrival[i, j])
                    arrival[i, j] = arrival[i, j] / max_cost * time_of_max_extent
                end
            end
        end
    end

    return arrival
end

#--------------------------------------------------------------------------------# _dijkstra_pass
function _dijkstra_pass(burned_mask, xs, ys, dx, dy, best_i, best_j,
                        model, t_mid, arrival_prev)
    ny, nx = size(burned_mask)
    dist = fill(Inf, ny, nx)
    dist[best_i, best_j] = 0.0

    pq = PriorityQueue{Tuple{Int,Int}, Float64}()
    push!(pq, (best_i, best_j) => 0.0)

    # 8-connected neighbor offsets: (di, dj)
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1))

    while !isempty(pq)
        (ci, cj) = popfirst!(pq).first
        d_curr = dist[ci, cj]

        for (di, dj) in neighbors
            ni, nj = ci + di, cj + dj
            (1 <= ni <= ny && 1 <= nj <= nx) || continue
            burned_mask[ni, nj] || continue

            # Euclidean edge distance [m]
            edge_dist = hypot(di * dy, dj * dx)

            # Compute edge cost
            if model isa FireSpreadModel
                # Direction-aware cost: use edge direction as fire front normal
                edge_nx = (dj * dx) / edge_dist
                edge_ny = (di * dy) / edge_dist
                mx = (xs[cj] + xs[nj]) / 2
                my = (ys[ci] + ys[ni]) / 2
                # Use previous-pass arrival time if available, else t_mid
                t_eval = if arrival_prev !== nothing && isfinite(arrival_prev[ci, cj])
                    arrival_prev[ci, cj]
                else
                    t_mid
                end
                speed = directional_speed(model, t_eval, mx, my, edge_nx, edge_ny)
                edge_cost = speed > 0 ? edge_dist / speed : Inf
            elseif model !== nothing
                # Isotropic callable model
                mx = (xs[cj] + xs[nj]) / 2
                my = (ys[ci] + ys[ni]) / 2
                t_eval = if arrival_prev !== nothing && isfinite(arrival_prev[ci, cj])
                    arrival_prev[ci, cj]
                else
                    t_mid
                end
                speed = model(t_eval, mx, my)
                edge_cost = speed > 0 ? edge_dist / speed : Inf
            else
                edge_cost = edge_dist
            end

            d_new = d_curr + edge_cost
            if d_new < dist[ni, nj]
                dist[ni, nj] = d_new
                if haskey(pq, (ni, nj))
                    pq[(ni, nj)] = d_new
                else
                    push!(pq, (ni, nj) => d_new)
                end
            end
        end
    end

    return dist
end

#--------------------------------------------------------------------------------# isochrones
"""
    isochrones(grid::LevelSetGrid, arrival_times::AbstractMatrix, times;
               reinit=IterativeReinit())

Extract fire perimeter snapshots at specified times from an arrival time field.

For each time `t` in `times`, creates a `LevelSetGrid` with `φ[i,j] = t - arrival_times[i,j]`
(negative = not yet burned, positive = already burned at time t), then optionally
reinitializes to restore the signed distance property.

# Arguments
- `grid` — The original `LevelSetGrid` (provides geometry: dx, dy, x0, y0, bc)
- `arrival_times` — Matrix from [`estimate_arrival_times`](@ref)
- `times` — Vector of times at which to extract perimeters
- `reinit` — Reinitialization method (default `IterativeReinit()`)

# Returns
`Vector{LevelSetGrid}` — one snapshot per requested time.

### Examples
```julia
T = estimate_arrival_times(grid, 1500.0, 1500.0, 100.0)
grids = isochrones(grid, T, [25.0, 50.0, 75.0, 100.0])
```
"""
function isochrones(grid::LevelSetGrid, arrival_times::AbstractMatrix, times;
                    reinit::AbstractReinitMethod=IterativeReinit())
    result = LevelSetGrid[]
    for t in times
        snap = LevelSetGrid(size(grid, 2), size(grid, 1);
                            dx=grid.dx, dy=grid.dy, x0=grid.x0, y0=grid.y0, bc=grid.bc)
        for j in axes(snap.φ, 2), i in axes(snap.φ, 1)
            at = arrival_times[i, j]
            # φ = arrival_time - t: negative (burned) when t > arrival_time
            snap.φ[i, j] = isinf(at) ? one(eltype(snap.φ)) : at - t
        end
        snap.t = t
        reinitialize!(snap, reinit)
        push!(result, snap)
    end
    return result
end

#--------------------------------------------------------------------------------# perimeter_to_grid
"""
    perimeter_to_grid(vertices, nx, ny; dx=30.0, dy=dx, padding=0.1)

Convert a polygon perimeter to a `LevelSetGrid`.

Creates a grid where `φ < 0` inside the polygon (burned) and `φ > 0` outside (unburned).
The grid domain is computed from the bounding box of the vertices with the specified
padding fraction.

# Arguments
- `vertices` — Vector of `(x, y)` tuples defining the perimeter polygon (should be closed,
  or will be implicitly closed)
- `nx, ny` — Grid dimensions
- `dx, dy` — Cell spacing [m]
- `padding` — Fraction of domain to add as buffer around perimeter

# Returns
A `LevelSetGrid` with the perimeter as the zero contour.

### Examples
```julia
# Diamond-shaped perimeter
verts = [(100.0, 0.0), (200.0, 100.0), (100.0, 200.0), (0.0, 100.0)]
grid = perimeter_to_grid(verts, 50, 50; dx=5.0)
```
"""
function perimeter_to_grid(vertices, nx, ny; dx=30.0, dy=dx, padding=0.1)
    vx = [v[1] for v in vertices]
    vy = [v[2] for v in vertices]

    xmin, xmax = extrema(vx)
    ymin, ymax = extrema(vy)
    xspan = xmax - xmin
    yspan = ymax - ymin

    x0 = xmin - padding * xspan
    y0 = ymin - padding * yspan

    grid = LevelSetGrid(nx, ny; dx=dx, dy=dy, x0=x0, y0=y0)
    xs = collect(xcoords(grid))
    ys = collect(ycoords(grid))

    n = length(vertices)
    for jj in eachindex(xs), ii in eachindex(ys)
        px, py = xs[jj], ys[ii]
        inside = _point_in_polygon(px, py, vertices, n)
        # Signed distance to polygon boundary
        min_dist = _point_to_polygon_distance(px, py, vertices, n)
        grid.φ[ii, jj] = inside ? -min_dist : min_dist
    end

    return grid
end

#--------------------------------------------------------------------------------# Point-in-polygon (ray casting)
function _point_in_polygon(px, py, vertices, n)
    inside = false
    j = n
    for i in 1:n
        xi, yi = vertices[i][1], vertices[i][2]
        xj, yj = vertices[j][1], vertices[j][2]
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end
    return inside
end

#--------------------------------------------------------------------------------# Point-to-polygon-edge distance
function _point_to_polygon_distance(px, py, vertices, n)
    min_dist = Inf
    j = n
    for i in 1:n
        xi, yi = vertices[i][1], vertices[i][2]
        xj, yj = vertices[j][1], vertices[j][2]
        d = _point_to_segment_distance(px, py, xi, yi, xj, yj)
        min_dist = min(min_dist, d)
        j = i
    end
    return min_dist
end

function _point_to_segment_distance(px, py, x1, y1, x2, y2)
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx^2 + dy^2
    if len_sq == 0
        return hypot(px - x1, py - y1)
    end
    t = clamp(((px - x1) * dx + (py - y1) * dy) / len_sq, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return hypot(px - proj_x, py - proj_y)
end

#--------------------------------------------------------------------------------# Validation Metrics
"""
    hausdorff_distance(grid_a::LevelSetGrid, grid_b::LevelSetGrid)

Compute the Hausdorff distance between two fire fronts.

Returns the maximum distance from any point on one fire front (`φ = 0`) to the
nearest point on the other front.  Uses the zero-crossing cells (cells adjacent
to a sign change in `φ`) from each grid.

Both grids must have the same dimensions and spacing.

### Examples
```julia
d = hausdorff_distance(simulated_grid, observed_grid)
```
"""
function hausdorff_distance(grid_a::LevelSetGrid, grid_b::LevelSetGrid)
    front_a = _front_points(grid_a)
    front_b = _front_points(grid_b)
    isempty(front_a) && isempty(front_b) && return 0.0
    (isempty(front_a) || isempty(front_b)) && return Inf

    d_ab = _directed_hausdorff(front_a, front_b)
    d_ba = _directed_hausdorff(front_b, front_a)
    return max(d_ab, d_ba)
end

function _front_points(grid::LevelSetGrid)
    ny, nx = size(grid)
    xs = collect(xcoords(grid))
    ys = collect(ycoords(grid))
    points = Tuple{Float64, Float64}[]
    for j in 1:nx, i in 1:ny
        φ_ij = grid.φ[i, j]
        is_front = false
        j < nx && φ_ij * grid.φ[i, j + 1] <= 0 && (is_front = true)
        i < ny && φ_ij * grid.φ[i + 1, j] <= 0 && (is_front = true)
        j > 1  && φ_ij * grid.φ[i, j - 1] <= 0 && (is_front = true)
        i > 1  && φ_ij * grid.φ[i - 1, j] <= 0 && (is_front = true)
        is_front && push!(points, (xs[j], ys[i]))
    end
    return points
end

function _directed_hausdorff(from, to)
    max_min = 0.0
    for (ax, ay) in from
        min_d = Inf
        for (bx, by) in to
            d = hypot(ax - bx, ay - by)
            min_d = min(min_d, d)
        end
        max_min = max(max_min, min_d)
    end
    return max_min
end

"""
    jaccard_index(grid_a::LevelSetGrid, grid_b::LevelSetGrid)

Compute the Jaccard index (intersection over union) of two burned areas.

Returns `|A ∩ B| / |A ∪ B|` where `A` and `B` are the burned regions (`φ < 0`).
A value of `1.0` indicates perfect agreement, `0.0` indicates no overlap.

### Examples
```julia
J = jaccard_index(simulated_grid, observed_grid)
```
"""
function jaccard_index(grid_a::LevelSetGrid, grid_b::LevelSetGrid)
    a = grid_a.φ .< 0
    b = grid_b.φ .< 0
    intersection = count(a .& b)
    union = count(a .| b)
    union == 0 && return 1.0
    return intersection / union
end

"""
    sorensen_dice(grid_a::LevelSetGrid, grid_b::LevelSetGrid)

Compute the Sorensen-Dice coefficient of two burned areas.

Returns `2|A ∩ B| / (|A| + |B|)`.  Similar to Jaccard but weights overlap more heavily.
A value of `1.0` indicates perfect agreement, `0.0` indicates no overlap.

### Examples
```julia
D = sorensen_dice(simulated_grid, observed_grid)
```
"""
function sorensen_dice(grid_a::LevelSetGrid, grid_b::LevelSetGrid)
    a = grid_a.φ .< 0
    b = grid_b.φ .< 0
    intersection = count(a .& b)
    total = count(a) + count(b)
    total == 0 && return 1.0
    return 2 * intersection / total
end

"""
    area_error(grid_a::LevelSetGrid, grid_b::LevelSetGrid)

Compute the relative burned area error between two grids.

Returns `(area_a - area_b) / area_b`.  A positive value means `grid_a` has more burned
area than `grid_b`.

### Examples
```julia
err = area_error(simulated_grid, observed_grid)
```
"""
function area_error(grid_a::LevelSetGrid, grid_b::LevelSetGrid)
    area_a = count(<(0), grid_a.φ) * grid_a.dx * grid_a.dy
    area_b = count(<(0), grid_b.φ) * grid_b.dx * grid_b.dy
    area_b == 0 && return area_a == 0 ? 0.0 : Inf
    return (area_a - area_b) / area_b
end

end # module
