# Approach

Status as of 2026-09-08.

## Goal

Rebuild Wildfires.jl as a coupled fire-atmosphere model equivalent in scope to
WRF-SFIRE, on the Oceananigans / Breeze.jl stack.

The previous implementation (`LevelSet`, `Rothermel`, `SpreadModels`,
`CellularAutomata`, PINN extension) is in `archive/src/`. Its physics is worth
porting; its grid and storage layers are not, since Oceananigans provides them.

## Why not WRF-SFIRE itself

WRF-SFIRE is two unequal pieces. SFIRE, the fire module, is on the order of 15k
lines of Fortran. WRF is roughly 1M lines accumulated over 25 years, mostly
physics parameterizations and registry-generated code. Reimplementing WRF is
neither tractable nor the useful part.

Reimplementing SFIRE against a Julia atmosphere is tractable, because Breeze.jl
already supplies the limited-area model — anelastic and compressible cores, LES
to mesoscale, terrain-following coordinates, nesting, GPU through
KernelAbstractions. The work reduces to the fire module and the coupling.

| Path | Estimate |
|------|----------|
| Full WRF-SFIRE clone | 30–55 person-months |
| Fire module + Breeze coupling | 8–12 person-months |
| Later: NumericalEarth component integration | 4–7 person-months |

To reproduce WRF-SFIRE's answers, running WRF-SFIRE is cheaper. The rewrite is
justified by three things it structurally cannot do: **differentiability**
(Enzyme/Reactant through the coupled system, enabling gradient-based parameter
estimation and 4D-Var — WRF-SFIRE has no adjoint), **GPU ensemble throughput**
for probabilistic forecasting, and **composability** with NumericalEarth's
Earth-system components.

## Architecture

### Dependencies

Oceananigans and Breeze are hard dependencies (Breeze was promoted from a
weakdep on 2026-09-04; the empty `WildfiresBreezeExt` was deleted). The raster
stack is an extension.

```
[deps]        Oceananigans, Breeze, KernelAbstractions, Proj, GeoFormatTypes, GeoInterface
[weakdeps]    Rasters -> WildfiresRastersExt (also needs ArchGDAL loaded)
[extras]      ArchGDAL (test only, for reading GeoTIFF through Rasters)
```

Proj is used only at setup on the CPU (`ProjectedCRS` origin, `to_grid`,
`to_lonlat`). It has no bearing on GPU support, and `GDAL_jll` depends on
`PROJ_jll` anyway, so dropping Proj.jl would not remove libproj from a process
that has ArchGDAL loaded.

Verified on Julia 1.12.7 with Oceananigans 0.110.19, Breeze 0.9.0,
Rasters 0.15.0, Proj 1.9.0.

### Two grids

Following SFIRE, the fire runs on a horizontal grid refined from the atmospheric
grid by an integer factor (typically 10). `fire_grid` derives it, so extents
match and every atmospheric cell contains exactly `refinement × refinement` fire
cells with coincident faces — e.g. an 8×6 atmosphere at 240 m gives a 64×48 fire
grid at 30 m.

Exact nesting makes field exchange a block operation rather than an
interpolation: aggregating fire fluxes up to the atmosphere is an exact sum,
conservative by construction. Hence `fire_grid` rejects a stretched atmospheric
grid instead of silently degrading to interpolation. One shared grid was
rejected because Rothermel spread on 200 m cells cannot resolve a fireline.

### GPU

GPU support through KernelAbstractions is a design requirement, not a later
port. Every operation on fire state must run on `architecture(grid)` without
change. The rules:

- **Fire state lives in Oceananigans `Field`s** on the grid's architecture.
  Never hold a plain `Array` of per-cell state.
- **Per-cell work is a `@kernel`** launched with
  `launch!(architecture(grid), grid, :xy, kernel!, args...)`. Read coordinates
  inside the kernel with `xnode(i, j, k, grid, ℓx, ℓy, ℓz)` and spacings with
  `Δxᶜᶜᶜ(i, j, k, grid)`; never index `xnodes(grid, ...)` or `interior(field)`
  from a CPU loop — that is scalar indexing on a `CuArray` and errors.
- **Setup-time code may be CPU-only** if it touches no device array. `set!`
  with a function is safe on GPU: Oceananigans evaluates it on a CPU copy of
  the grid, then copies (`Fields/set!.jl`, `set_to_function!`). Functions
  passed to `set!` must be thread-safe, because the CPU evaluation is a
  multithreaded KernelAbstractions kernel.
- **Kernel arguments must be isbits or `Adapt`-able.** Closures over `Array`s,
  `Proj.Transformation`s, or `Raster`s cannot be passed to a kernel.
- **Verify with `GPU()`** where hardware is available, and otherwise keep
  kernels free of anything that would not compile for the device.

`ignite!` (`src/fire_model.jl`, `_ignite!`) is the reference kernel: API
function validates and converts scalar arguments to `FT = eltype(grid)`,
`launch!(architecture(grid), grid, :xy, _ignite!, fields..., grid, scalars...)`,
then `fill_halo_regions!`. Verified on Metal (Float32) as well as CPU; Metal
is the stricter target because it has no Float64, so a stray `Float64`
literal (`fill!(parent(φ), Inf)` was one) fails there where CUDA would only
slow down.

### Enzyme

Differentiability is a secondary goal: keep the code Enzyme-compatible, but
where a rule here conflicts with GPU support or clarity, GPU wins.

Two AD paths exist upstream and both consume the same kernel code. Breeze's
supported path is **Reactant** (`ReactantState` architecture, Enzyme applied
to the compiled XLA program; all of Breeze's AD tests are under
`test/reactant/`). Oceananigans also differentiates **natively** with Enzyme
on CPU (`test/test_enzyme.jl` reverse-differentiates `time_step!` of a
`HydrostaticFreeSurfaceModel` with `Enzyme.set_runtime_activity(Reverse)`);
its `OceananigansEnzymeExt` marks grid metrics and utilities inactive, and
the custom `launch!` rules are commented out, so KernelAbstractions CPU
kernels are differentiated directly. Writing for both means:

- **Fixed trip counts.** No `while` or data-dependent loop bounds in the
  differentiated path. Reinitialization sweeps, iterative solves, and
  substepping take a fixed iteration count (Breeze's own saturation solver
  does this for the same reason — a data-dependent XLA `while` has
  pathological reverse-mode).
- **`ifelse`, `min`, `max`, `clamp` over `if` on field values inside kernels.**
  Reactant cannot trace a scalar branch on a traced value; Enzyme handles
  either, but `ifelse` serves both.
- **Sentinels stay out of arithmetic.** `φ` and `t_ignition` initialize to
  `Inf`. Any diagnostic derived from them (burned fraction, heat release) must
  mask un-ignited cells with `ifelse(isfinite(t_ignition), …, zero(FT))` before
  the expression, never compute `Inf - Inf` and fix it after.
- **Parameters are arguments, not globals.** Anything a user might estimate —
  Rothermel coefficients, fuel-table values, wind-adjustment factors, level-set
  reinitialization strength — lives in a struct passed through `FireModel`, so
  it can be `Duplicated`/`Active`. Fuel-model integer codes are inherently
  non-differentiable; the per-code table values are where gradients go.
- **No foreign calls in the differentiated path.** Proj, GDAL, and file I/O
  are setup only, which the GPU rules already require.
- **Type-stable kernels.** Enzyme fails on dynamic dispatch (`Any`) with
  "no augmented forward pass found"; the fix is the same as for GPU.
- **No `error`/`throw` on field values inside kernels.** Validate at setup.
- **Verify as kernels land.** For each new `time_step!`-path kernel, add a
  Reactant compile test in the style of Breeze's `test/reactant/`, and where
  the problem is small enough, an `Enzyme.gradient` of a scalar loss with
  respect to one parameter on CPU. Discontinuities (ignition indicator,
  upwind switches) give zero or one-sided gradients by design; document
  which, rather than smoothing them without cause.

### Field types

Surface fields on the atmospheric grid are `Field{Center, Center, Nothing}` —
`Nothing` in the vertical means one value per column, giving `(Nx, Ny, 1)`
instead of a volume field's `(Nx, Ny, Nz)`. This is what Breeze's
`materialize_terrain!` builds internally, and the fire state uses the same
locations even though the fire grid's `Flat` vertical already collapses it, so
terrain and fire state share a type on both grids.

### Georeferencing

Oceananigans grids carry no CRS, so georeferencing is a separate object passed
alongside the grid rather than a wrapper around it. `ProjectedCRS` holds an EPSG
code and the projected coordinate of the grid's `(0, 0)`.

A projected `RectilinearGrid` in meters was kept over Oceananigans'
`LatitudeLongitudeGrid`. Breeze supports terrain and compressible dynamics on
lat-lon grids, but its anelastic pressure solver is
`FourierTridiagonalPoissonSolver`, which is rectilinear-only; lat-lon would
force compressible dynamics. Projected grids also match WRF-SFIRE and keep the
fire's spread math Euclidean.

Terrain enters as a closure. `raster_topography(raster, crs)` returns
`(x, y) -> h` for Breeze's `materialize_terrain!(grid, topography)`, which
allocates the surface field, evaluates the closure at cell centres via `set!`,
and fills halos — so Wildfires does not own that loop. The raster is warped
once into the grid's CRS with `Rasters.resample` (GDAL, bilinear) and the
closure is a pure bilinear lookup with the origin subtracted. Warping up front
rather than transforming each query point keeps Proj out of the closure, which
`set!` evaluates from multiple threads; `Proj.Transformation` is not
thread-safe. Any source CRS works — LANDFIRE Albers, HRRR Lambert conformal.
Bilinear rather than nearest because `materialize_terrain!` finite-differences
the elevation field for slopes, and stair-stepped slopes would corrupt both the
coordinate metric and the fire's slope response.

All Proj transformations use `always_xy = true`. Many CRS definitions declare
latitude first; without it, coordinates come back swapped.

## Progress

Each step was verified before moving on. Test counts are cumulative.

| Step | Delivered | Tests |
|------|-----------|-------|
| 1 | Package skeleton: deps, extension stubs, loadable module | 1 |
| 2 | `ProjectedCRS`, CRS transforms, `elevation_field` | 21 |
| 3 | `fire_grid`, `FireModel` | 43 |
| 4 | `ignite!` | 57 |
| 5 | Breeze hard dep; `elevation_field` → `raster_topography` closure | 56 |
| 6 | `ignite!` as a kernel; `FT(Inf)` sentinels; Metal + Enzyme verified | 56 |
| 7 | `advance!`: Godunov + Heun level-set step with prescribed speed | 70 |
| 8 | Rothermel: `FuelClasses`, `FuelModel`, `FuelBed`, `spread_rate`, NFFL + SB40 tables | 330 |
| 9 | `raster_sampler` (nearest or bilinear); `spread_rate!` speed field along the front normal; `examples/marshall/marshall_fire.jl` | 364 |
| 10 | WUI: `Structures` from footprints, Hamada urban spread in `spread_rate!`, `wind_adjustment`, structure ignition and heat release | 430 |
| 11 | `HuygensEllipse` front shape, `FuelMap` device fuel table, `raster_time_series` wind, `polygon_coverage`, polygon `ignite!`; eight-hour Marshall run | 481 |
| 12 | HRRR into Breeze: `raster_column_series`, `lateral_sponge`, `relaxation_forcing`; `examples/marshall/atmosphere.jl` | 509 |

```julia
crs   = ProjectedCRS(-105.182, 39.9575)        # UTM 13N, origin at that point
h     = raster_topography(Raster("elevation.tif"), crs)
materialize_terrain!(grid, h)                  # Breeze

model = FireModel(fire_grid(atmosphere; refinement = 8))
ignite!(model, 0.0, 0.0; radius = 100.0)
```

**Step 2** (`src/georeference.jl`, `ext/WildfiresRastersExt.jl`) verified end to
end on `docs/data/marshall/elevation.tif`: the origin round-trips exactly, the
sampled field (1629.6–1805.9 m) lies inside the DEM range (1612–1841 m), no
NaNs, and `znode` at the bottom face equals the terrain height to floating
point. `to_grid` / `to_lonlat` build a transformation per call for one-off
points; `to_grid_transform` / `from_grid_transform` return a reusable closure
for bulk work.

**Step 3** (`src/fire_model.jl`). `FireModel` carries `grid`, `clock`, and two
prognostic fields: `φ`, the level set (`φ < 0` burning or burned, `φ > 0`
unburned, zero contour the perimeter), and `t_ignition`, when each cell ignited,
for post-frontal fuel consumption and heat release. Burned fraction is not
stored — SFIRE computes it diagnostically from ignition time and the fuel's
burn-out timescale, and so will this. Both fields initialize to `Inf`, meaning
no fire anywhere; zero would place the front everywhere, and advancing an
un-ignited model is undefined. `Oceananigans.fields(model)` returns
`(; φ, t_ignition)`, so output writers work without extra glue.

**Step 4.** `φ` is set to the exact signed distance to the ignition circle, so
`|∇φ| = 1` and no reinitialization is needed before the first advection step
(asserted to within 2% more than four cells from the center singularity).
Ignitions combine by pointwise minimum: the minimum of two signed distance
functions is the signed distance to their union, so repeated ignitions compose
and are order-independent, both asserted. `t_ignition` keeps the earlier time,
so a burning cell is not reset by a nearby later ignition. Coordinates are grid
meters — a real ignition point is
`ignite!(model, to_grid(crs, lon, lat)...; radius = 100)`.

**Step 5.** `elevation_field(grid, raster, crs)` duplicated Breeze's
`terrain_height_field` (allocate `Field{Center, Center, Nothing}`, evaluate at
centres, fill halos), so it was replaced by `raster_topography(raster, crs)`,
which returns the closure `materialize_terrain!` expects and takes no grid.
The test asserts `znode` at the bottom face equals the closure evaluated at
cell centres, and that the same closure `set!`s onto the fire grid. The Rasters
ext no longer imports Oceananigans. `to_grid_transform` /
`from_grid_transform` are no longer used internally.

**Step 6.** `ignite!` rewritten as a KernelAbstractions kernel after the CPU
loop failed on Metal with `Scalar indexing is disallowed`, and
`FireModel`'s `fill!(…, Inf)` changed to `FT(Inf)` after Metal rejected the
Float64 literal. Checked outside the test suite (no GPU in CI): on
`GPU(Metal.MetalBackend())` with Float32, `FireModel`, two `ignite!` calls
with Float64 arguments, and `set!(field, raster_topography(dem, crs))` all
run and produce `MtlArray`s with the expected values; on CPU,
`Enzyme.autodiff(Reverse, …)` through `ignite!` gives
`d(Σφ)/d radius = -1024.0` against a finite difference of `-1023.9999997`.
The kernel uses `ifelse` for the ignition-time update so it also traces
under Reactant.

**Step 7.** `advance!(model, Δt; speed)` solves `∂φ/∂t + speed |∇φ| = 0`
with the Godunov upwind Hamiltonian (Osher & Fedkiw §6.2 max form,
`max(max(D⁻,0), -min(D⁺,0))` per direction) and Heun's method, the
WRF-SFIRE scheme. `speed` is a number or a field on the fire grid; the
model owns one `scratch` field for the first stage, so the step allocates
nothing. Differences come from Oceananigans' `∂xᶠᶜᶜ` / `∂yᶜᶠᶜ`, so the
stencil is one cell and boundary behaviour is the field's BCs — the
default no-flux fills halos zero-gradient, which is the zero-Neumann the
old code implemented by hand. `t_ignition` is set by linear interpolation
of the zero crossing within the step; the divisor is `ifelse`-guarded so
the unselected branch is finite. Validation against the expanding circle
(uniform speed, exact solution `r - R - F t`): φ is within one cell in
the far field, `t_ignition` within one cell's transit of `(r - R)/F`,
and the burned area 5% low at `R + F t = 200 m` on a 10 m grid. That lag
is the first-order upwind error on a cone — Godunov's backward difference
underestimates the gradient of a convex profile, integrating to
`(Δx/2) ln(r_end/r_start)` ≈ 7 m here — and the reason a higher-order
scheme is still on the list. Inside radius `F t` of the centre φ is wrong
by up to `F t`: the local minimum has zero upwind gradient, so it stays
fixed and the error spreads outward. It does not affect the front.
Metal (Float32, field speed) matches CPU to 0.0; Enzyme reverse through
ten steps gives `d(Σφ)/d speed = -9633.2438115` against a finite
difference of `-9633.2438123`.

**Step 8** (`src/rothermel.jl`). Rothermel (1972) ported from
`archive/src/Rothermel.jl` and split into `FuelBed`, which precomputes every
fuel-only quantity (area weights, net loads, reaction velocity Γ, propagating
flux ξ, the wind and slope coefficients), and `spread_rate(bed, moisture,
wind, slope)`, which evaluates only the moisture damping, heat sink, and
wind/slope factors per call — the per-cell cost when this runs in a kernel.
Interface is SI (midflame wind m/s, slope rise/run, result m/s); `FuelModel`
keeps the tables' US customary units so entries can be checked against the
publications. Two changes from the archive: Albini's (1976) size-bin net-load
weighting, which matters only for categories with two classes in one bin
(GS, SH9, TU1, TU3 live fuel), and GR6's 9000 BTU/lb heat content. Negative
wind is treated as calm and negative slope as flat (as in SFIRE), so the
arguments can be components along the spread direction. All divisors and
power bases are guarded with `_pos` so an empty fuel bed gives finite
intermediates, not `Inf` masked after the fact.

Validated against the Forest Service `behave` library (BehavePlus core, via
pyrothermel 0.1.4) for all 53 models under two scenarios: relative error
≤ 1e-4 for SB40 and ≤ 2.5e-3 for NFFL (behave's NFFL loads round-trip
through kg/m²) wherever behave's wind limit is inactive. Every remaining
discrepancy was reproduced exactly by imposing `U ≤ 0.9 I_R` (Rothermel
1972), which behave applies by default and this code does not — see Next.
Anderson (1982) Table 1 values are 5–30% off from behave for the same inputs
and were not used. Out of CI: Enzyme reverse gradients with respect to
moisture, wind, slope, and a fuel load (through `FuelBed` construction) match
finite differences to 1e-9; the gradient at exactly zero wind is 0 by
construction (the `max(U, 1e-10)` clamp). A Float32 Metal kernel calling
`spread_rate` per element matches CPU Float64 to 6e-4 relative.

**Step 9** (`ext/WildfiresRastersExt.jl`, `src/spread.jl`, `examples/`).
`raster_sampler(raster, crs; method = :near | :bilinear, fill_value)` is the
general form of `raster_topography`, which is now the bilinear special case;
`method` applies to both the GDAL warp and the lookup. Two corrections to the
sampler: GDAL lookups hold cell *starts* (`Intervals(Start())`), so values are
now placed at cell centres (the old bilinear sampled half a pixel off), and the
outer half-cell strip clamps to the edge instead of returning `fill_value`.
Tested on a synthetic UTM raster, where the warp is the identity.

`spread_rate!(speed, model, bed, moisture; wind = (u, v), slope = (∂x_h, ∂y_h))`
is the SFIRE formulation: central-difference normal `n = ∇φ/|∇φ|` (`_pos`
makes it zero, not NaN, where `∇φ = 0`), wind and slope resolved as `U·n` and
`∇h·n`, negatives treated as calm/flat by `spread_rate`. Components are
numbers or fields on the fire grid. One `FuelBed` for the whole domain until
the fuel table lands. `FuelBed{T}(bed)` converts between float types so a
Float64 table serves a Float32 grid. Verified: downwind cell at the head-fire
rate, upwind cell at exactly the no-wind rate, field and number arguments give
identical results, Float32 grid runs. Out of CI, on `GPU(Metal.MetalBackend())`
with Float32: number and field arguments both match CPU to 1e-6, and ten
coupled `spread_rate!`/`advance!` steps match to 3e-5. (`set!(field, 5.0)` on
a Metal field fails on the Float64 literal; pass `eltype(grid)(5)`.)

`examples/marshall/marshall_fire.jl` is the driver: LANDFIRE fuel codes (nearest) mask
non-burnable cells by multiplying `speed` by a 0/1 field, LANDFIRE slope and
aspect (nearest, since aspect wraps) give `∇h` directly, HRRR 10 m wind is
sampled bilinearly per 15 min snapshot with an inline Albini–Baughman
adjustment factor. Findings from the first runs:

- **Flank spread is ~1% of head spread.** Fuel model 2 at 6% moisture gives
  `R₀ = 0.016 m/s` against `R ≈ 1.5–2.2 m/s` downwind, so an hour produces a
  strip one ignition-diameter wide. That is what the normal-projection
  formulation does (SFIRE relies on fire-induced winds to widen it); FARSITE,
  ELMFIRE, and the archived CA code use Huygens ellipses, whose flank rate is
  head rate over the length-to-breadth ratio, roughly ten times larger here.
  Which to use is an open design decision.
- **Scattered non-burnable pixels stall a narrow head.** With the flanks that
  slow, every urban pixel casts a shadow the front cannot fill, so the 34%
  urban fraction east of the ignition halves the head's progress in the first
  15 min and stops it soon after. The ignition pixel itself is urban (the
  Twelve Tribes property); a 50 m ignition never leaves it, 100 m does.
- Burned area after one hour: 0.2 km² against a 24.4 km² final perimeter.

**Step 10** (`src/structures.jl`, `src/hamada.jl`, `src/spread.jl`). The
wildland-urban interface follows the two-layer design: footprints are
first-class objects with their own state, and ELMFIRE's per-cell rasters are
derived from them so the level set remains the only front.

`Structures(geometries, crs, grid; classes, class, radius, subsamples)` takes
any GeoInterface polygon collection (GeoJSON feature collections included),
reprojects the exterior rings with `to_grid_transform`, and keeps, per
structure on the device, centroid, footprint area, class, `t_ignition`, and
the footprint cells in compressed-row form. Coverage is sampled on a 4 × 4
lattice per cell with even-odd ray casting; no Rasters dependency, so it
lives in core with GeoInterface as a hard dependency (added 2026-09-08).
Per-cell fields are `footprint_fraction` and neighbourhood means over the
structures within `radius` (60 m) of the cell centre: `plan_dimension`
(√area), `separation` (edge-to-edge estimate to the nearest other structure
via bins of size `radius`, capped at `radius`), and `nonburnable_fraction`.
These match ELMFIRE's `BLDG_AREA`, `BLDG_SEPARATION_DIST`,
`BLDG_NONBURNABLE_FRAC`, `BLDG_FOOTPRINT_FRAC`; `BLDG_FUEL_MODEL` is the
`BuildingClass` (fire-resistant flag plus ELMFIRE's heat-release curve:
growth, plateau, decay, peak per m² of footprint, defaults from ELMFIRE's
building fuel model 1).

`hamada_rates` is a port of ELMFIRE's `HAMADA` subroutine (Hamada 1951 as
used in HAZUS): downwind, crosswind, and upwind rates from open wind, plan
dimension, separation, and fire-resistant fraction, with the HAZUS blend
below 10 m/s and the small-extent fallback, written with `_pos`/`ifelse` so
it runs in kernels. Checked against a hand calculation (0.27 m/s downwind at
15 m/s for 15 m buildings 10 m apart). One property worth knowing: the front
advances one block pitch per building burn time, so wider separation raises
the rate; only the fire-resistant fraction lowers it. `ellipse_speed` is the
support function of the Huygens ellipse with those three rates, which is the
normal speed the level set needs, and is the same machinery a
length-to-breadth ellipse for Rothermel would use.

`spread_rate!` now takes the *open* wind and applies `wind_adjustment(bed)`
(Albini and Baughman 1979, unsheltered, 0.36 for 1 ft beds) to the
vegetation term, because Hamada needs the open wind and one call cannot take
two conventions. With `structures` the kernel adds the urban term in cells
whose `plan_dimension` is positive and keeps the larger of the two rates; a
cell with a house and grass burns at whichever is faster.

`update_structures!` reduces the level set's `t_ignition` over each
structure's footprint cells (a data-dependent loop bounded by footprint
size, the one such loop in the time-step path). `heat_release` evaluates
area × class curve from `t_ignition`, masked by `isfinite` before any
arithmetic, so structure flux to Breeze follows the same pattern as the
planned vegetation burn-out. Out of CI on Metal (Float32): `Structures`
builds device vectors and fields, twenty coupled steps with the urban term
match CPU speed to 4e-5, and structure ignition times and heat release match
exactly. The one Metal-only failure found was the derived fields being
`set!` from Float64 arrays; they are converted to `eltype(grid)` first.

Marshall: `download_buildings.jl` pulls 12,709 OSM footprints for the tile
extent through OverpassAPI.jl (RP1's client; the docs env `dev`s
`~/.julia/dev/OverpassAPI` because v0.1.0 has two uncommitted fixes there:
Overpass rejects its multipart POST and its second settings statement with
HTTP 400 — release v0.1.1 and switch back to the registry); 10,892 fall on
the fire grid. With the urban term
the one-hour burned area doubles to 0.4 km² and 46 structures ignite, heat
release 0.19 GW at 19:00 UTC. Every building is the default class because
no construction attributes are available; the county damage inspection
(1,084 destroyed) is the calibration target once it is in `data/`.

Not done from the design: the structure-to-structure exposure model
(ELMFIRE's UMD-UCB direct-flame-plus-radiation kernel, or SWUIFT's
probabilities). With Hamada spreading fire through urban cells it would be a
second, competing spread mechanism; it belongs either as a replacement for
Hamada or as the ember/independent-ignition path, and its parameters need
calibration data. The per-structure state and neighbour bins it needs exist.

**Step 11** (`src/spread.jl`, `src/polygons.jl`, `src/georeference.jl`).
The four items that stood between the example and a skill number.

- **Front shape.** `spread_rate!` takes `shape = HuygensEllipse()` (default)
  or `NormalProjection()` (the SFIRE formulation of Step 9). The ellipse
  path converts the slope to the midflame wind giving the same Rothermel
  slope factor, `(C_s tan²θ / C_w)^{1/B}`, adds it as a vector to the
  adjusted wind, evaluates the head rate in that effective wind, takes the
  length-to-breadth ratio from `length_to_breadth` (Anderson 1983 as in
  FARSITE and ELMFIRE, capped at 8), and returns the ellipse's support
  function via `ellipse_speed`. At 5 m/s open wind the flank is 21% of the
  head; the normal projection gave the no-wind rate, 1–7% of the head
  depending on fuel and moisture. Tested: head equals the Rothermel rate, flank and
  back match the ellipse axes, the upslope head equals the Rothermel slope
  rate exactly (the equivalent wind reproduces the factor), opposing wind
  and slope slow the head.
- **Fuel table.** `FuelMap(grid, code, table)` builds one `FuelBed` per
  model in `NFFL`, `SB40`, or any `FuelModel` list, stores the beds as a
  device vector (a tuple would exceed the CUDA kernel-argument limit at
  40 × 320 B), and a per-cell index field. Codes not in the table map to the
  new `NONBURNABLE` model, whose bed spreads at zero by construction, so the
  example's mask multiply and second urban-only call are gone. The wind
  adjustment factor is now evaluated per cell from the cell's bed.
- **Wind time series.** `raster_time_series(rasters, times, grid, crs)`
  returns an Oceananigans `FieldTimeSeries` (`Clamp` outside the range), and
  `spread_rate!` reads any `FieldTimeSeries` component at
  `model.clock.time` inside the kernel through `fts[i, j, 1, Time(t)]`,
  which the GPU-adapted series supports since `times` survives adaptation.
  `download_hrrr.jl` now covers 17:00 to 02:00 UTC (37 snapshots).
- **Polygons.** `polygon_coverage` rasterizes any GeoInterface polygons to a
  coverage field, and `ignite!(model, geometries, crs)` sets `φ` to the exact
  signed distance to the exterior rings (CPU, cells × edges, threaded) by
  pointwise minimum, so an observed perimeter both ignites and
  re-initializes. The rasterization helpers moved from `structures.jl` to
  `polygons.jl` and `Structures` uses the shared `_footprints`.

Marshall, eight hours, 18:00 to 02:00 UTC, 2 min 47 s on the CPU:

| | Model | Observed |
|---|---|---|
| Burned area | 7.6 km² | 24.3 km² |
| Hit / miss / false alarm | 6.1 / 18.2 / 1.5 km² | |
| Critical success index | 0.24 | |
| Structures ignited | 208 of 10,892 | 1,084 destroyed |
| Peak structure heat release | 0.59 GW | |

`examples/marshall/figures.jl` (included by the example, CairoMakie) writes
`inputs.png` (terrain, fuel, structures, wind), `arrival_time.png`,
`timeseries.png`, and `propagation.gif`, the last built from the arrival-time
field alone since burned-at-`t` is `t_ignition ≤ t`. The arrival-time map
shows the fire running east as three narrow tongues along the 3 km HRRR wind,
which is due west at the ignition, while the observed perimeter's southern
lobe (the real spread was east-southeast) is missed entirely; the late
yellow halo around the ignition is isotropic spread after the wind drops.
Growth is close to 1 km² per hour throughout, including after the wind drops
to 5 m/s at 00:00 UTC, whereas the real fire made most of its run in the
first four hours and stopped under snow. Of what burns, 80% lies inside the
observed perimeter, so the direction is right and the rate is a factor of
three low.

**Urban fuel imputation** (same day). LANDFIRE classes a third of the domain
as 91, developed, which the FBFM13 table makes non-burnable, so the
vegetation front stopped at every street and only Hamada (~0.3 m/s) carried
it through the towns. `FuelMap(...; remap = (91 => 1,))` treats developed
cells as short grass — cured lawns and plantings — as WRF-Fire WUI studies
of the Camp and Marshall fires do. Effect over the same eight hours:

| | Non-burnable 91 | 91 as short grass | Observed |
|---|---|---|---|
| Burned area | 7.6 km² | 16.5 km² | 24.3 km² |
| Hit / miss / false alarm | 6.1 / 18.2 / 1.5 | 11.3 / 13.0 / 5.2 | |
| Critical success index | 0.24 | 0.38 | |
| Structures ignited | 208 | 1,662 | 1,084 destroyed |
| Peak structure heat release | 0.6 GW | 12 GW | |

The front now runs through Superior and Louisville, and the remaining miss
is the southern lobe (the observed east-southeast spread) and the north edge.
Structures ignited now exceed structures destroyed: every structure the front
touches ignites, with no ignition probability or hardening in the
level-set-to-structure exchange, so a per-class ignition probability (ELMFIRE's
`P_IGNITION`, SWUIFT's building types) is the next structure-side item. The
short-grass choice is a placeholder: fuel model 1 at 6% moisture is the
fastest of the FBFM13 set, and the false-alarm area tripled. Remaining
candidate causes for the miss: the sustained HRRR wind at 3 km against gusts
near 50 m/s (gust rasters are downloaded and unused), the wind direction over
this terrain, no spotting, no fire-induced winds, and the moisture guess.

**Step 12** (`src/atmosphere.jl`, `docs/data/marshall/download_hrrr_levels.jl`,
`examples/marshall/atmosphere.jl`). The first of the three coupling steps:
the analysis as initial state and lateral forcing of a Breeze atmosphere, the
role WPS and `real.exe` play for WRF.

- `download_hrrr_levels.jl` pulls the hourly `wrfprsf00` analyses from 17:00
  to 02:00 UTC by byte range: HGT, TMP, SPFH, UGRD, VGRD on the 29 isobaric
  levels from 1000 to 300 hPa, one multi-band GeoTIFF per variable and hour,
  plus surface pressure and terrain height. GDAL's GRIB driver reports
  temperature in °C, which cost one run.
- `raster_column_series(values, heights, times, grid, crs)` samples every
  band at every column and interpolates linearly in geopotential height onto
  the column's cell centres of the terrain-following grid, holding the end
  levels constant beyond their range. HRRR's below-ground levels supply the
  values under the 1.7 km terrain. It returns a 3D `FieldTimeSeries`.
- `lateral_sponge(grid; width, ramp)` is the Davies mask, `relaxation_forcing(name, target, rate; mask)`
  an Oceananigans discrete forcing reading a `FieldTimeSeries` target at the
  clock time inside the kernel (the `Val`-typed field name keeps it
  type-stable); Breeze wraps it in `SpecificForcing` under the specific key.
  Tested on a periodic anelastic box: uniform relaxation follows the
  exponential, the sponge leaves the interior untouched, and the target is
  read at the Runge–Kutta stage times, which the first test did not expect.
- The example builds `CompressibleDynamics` (split-explicit, `UpperSponge`,
  `SlopeInsideInterpolation`, the configuration Breeze documents on terrain)
  with the analysis-mean θ profile and sea-level pressure from the surface
  fields, `SmagorinskyLilly`, WENO5, and relaxation of `u`, `v`, `θ` over the
  outer eight cells at 1/600 s; sets the initial state from the 18:00 UTC
  analysis (velocities averaged onto faces) with `compute_reference_state`;
  and runs one hour, seven minutes of wall time for 72k cells on the CPU.

Finding: with Breeze's default impenetrable side walls the through-flow
cannot enter, and the level-1 wind falls from 18 to 8 m/s within ten minutes
while the analysis holds 19, with max |w| reaching 13 m/s as the flow diverts
(`figures/atmosphere_wind.png`). Relaxation alone does not make a lateral
boundary. Breeze 0.9's acoustic substepping does honour
`NormalFlowBoundaryCondition` on `ρu`/`ρv`; one attempt with discrete-form
values `ρᵈ × analysis` at the walls and `PerturbationAdvection(600, 60)`
built and initialized but produced a negative thermodynamic state on the
first step. Not pursued further in this session. Moisture is initialized
but not relaxed (`qᵛ` is not among the forcing-visible fields; relax `ρqᵛ`
instead), and there is no surface layer, so the lowest level feels no drag.

## Next

Unblocked, in rough dependency order:

- **Open lateral boundaries for the Breeze run** — see Step 12. Get
  `NormalFlowBoundaryCondition` on `ρu`/`ρv` working with the compressible
  core: check the face index and density used for the target momentum, try
  the default `PerturbationAdvection()` (instant inflow, free outflow), and
  compare with NumericalEarth 0.7's `parent_boundary_conditions`, which
  wraps the series in `Interpolated`. Until then the atmosphere example
  demonstrates ingestion, not a usable wind.
- **Close the factor of three** — see the Step 11 table and figures. Try in
  order: drive with HRRR gusts (or a blend) instead of the sustained wind;
  check the wind direction against the observed east-southeast spread (a
  finer wind product, or the coupled atmosphere once its boundaries work);
  calibrate moisture and the Hamada coefficients against the perimeter with
  the overlap statistics; spotting.
- **Structure ignition probability** — the front ignites every structure it
  touches (1,662 against 1,084 destroyed). A per-class probability or
  hardening factor in `update_structures!`, calibrated on the county damage
  inspection.
- **Structure exposure model** — see Step 10. Decide whether ELMFIRE's
  UMD-UCB kernel or SWUIFT replaces Hamada or supplements it for embers.
- **Structure flux into Breeze** — `heat_release` per structure onto the
  atmosphere cell of its centroid, alongside the vegetation flux.
- **Building classes from data** — map OSM `building` tags, FEMA USA
  Structures, or assessor parcels to `BuildingClass`; Boulder County's
  damage inspection for calibration.
- **Decide the wind-speed limit.** BehavePlus/FARSITE/FlamMap impose
  Rothermel's `U ≤ 0.9 I_R`; Andrews et al. (2013) recommend against it and
  `spread_rate` omits it. Reproducing FARSITE/ELMFIRE runs needs it as an
  option; WRF-SFIRE caps wind speed differently. Cheap to add as a `FuelBed`
  or `spread_rate` argument.
- **Dynamic fuel models** — SB40 GR/GS/SH9/TU1/TU3 transfer cured herbaceous
  load from live to dead as a function of herb moisture; static loads for now.
- **Overlap statistics as a package function** — the example computes hit,
  miss, false alarm, and the critical success index inline; worth a
  function once a second case study exists.
- **Higher-order level set** (WENO3/5 + SSP-RK3) if the first-order lag
  matters; `fire_grid`'s default halo of 3 already accommodates WENO5.
  Reinitialization is not needed while speed is uniform, and may not be
  needed at all if `t_ignition` rather than `φ` drives the physics.

Further out: heat and moisture flux injection into Breeze (via
`forcing_interface.jl` or `BoundaryConditions/bulk_scalar_fluxes.jl`),
prognostic fuel moisture, and the NumericalEarth component wrapper.

## Known constraints

- **`z_top` is absolute altitude.** Marshall terrain sits near 1700 m, so a 4 km
  model top leaves only ~2.3 km of atmosphere — thin for a fire plume, with the
  sponge layer close to it. Set `z_top` relative to terrain elevation for real
  domains.
- **`TwoLevelDecay` over `LinearDecay` for real terrain.** LinearDecay keeps
  small-scale topography imprinted on coordinate surfaces up to the lid,
  generating spurious mixing aloft. LANDFIRE-resolution terrain is that case.
- **`EarthSystemModel` has fixed component fields** (`radiation`, `atmosphere`,
  `land`, `sea_ice`, `ocean`). A first-class `fire` component needs an upstream
  PR; the alternative is riding inside `land` via `property_providers.jl`.
- **Timescale mismatch with the Earth-system loop.**
  `time_step_earth_system_model.jl` steps every component with one shared `Δt`.
  Fire wants seconds and tens of meters; the exchange grid is kilometers, and
  there is no subcycling in that loop. Raise with the NumericalEarth maintainers
  before committing to the component path.
- **`regrid_topography` assumes a lat-lon target grid.** It builds intermediate
  `LatitudeLongitudeGrid`s and indexes the target as `(Nλ, Nφ)`. Hence our own
  Proj-based DEM sampling onto a metric `RectilinearGrid`.
- **Breeze and NumericalEarth move quickly.** Verify APIs against `main` rather
  than relying on notes.

## Repository state

The previous implementation lives under `archive/` as of 2026-09-04: `src/`,
`test/`, `docs/` (chapters, gifs, `generate_hero.jl`, troubleshooting page),
`ext/` (GPU, Makie, NeuralPDE, PINN — the `ext/` copies had already been
gutted to stubs, so the archived files are the last full versions from git),
and `CHANGELOG.md`. Nothing in `archive/` is loaded or tested. `Terse` is
still a dependency but unused.

`archive/ext/WildfiresGPUExt/*_kernel.jl` are old KernelAbstractions kernels
against the archived `Grid` type — the same idiom the new fire kernels use,
against different storage.

`archive/docs/refactor-sketch.qmd` describes the pre-Breeze architecture. Its
`Grid` and `Environment` layers are superseded by Oceananigans; its
`FireShape` and `RateOfSpread` layers still stand.

`docs/` now holds only `index.qmd`, `api.qmd`, and the resources pages; its
Project.toml was rebuilt with Pkg (ArchGDAL, JSON3, LocalCoverage,
Oceananigans, Rasters, Wildfires).

## Conventions

- Source files defining a module are CamelCase (`Wildfires.jl`,
  `WildfiresRastersExt.jl`); files merely `include`d are lowercase
  (`georeference.jl`, `fire_model.jl`).
- Build in small increments, surfacing design forks rather than deciding them
  silently.
- Reference implementation for spread physics: the Forest Service `behave`
  C++ library through `pip install pyrothermel` in a scratch venv
  (`PyrothermelRun(FuelModel.from_existing(code, units_preset = "us_standard"),
  MoistureScenario(...), wind, wind_input_mode = "direct_midflame", slope = deg)`,
  set `is_dynamic = false` for static loads).
