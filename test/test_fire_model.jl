@testset "fire_grid" begin
    atmosphere = RectilinearGrid(size = (8, 6, 4), halo = (3, 3, 3),
                                 x = (-960.0, 960.0), y = (-720.0, 720.0), z = (0.0, 400.0),
                                 topology = (Bounded, Bounded, Bounded))
    fg = fire_grid(atmosphere; refinement = 8)

    @test size(fg) == (64, 48, 1)
    @test topology(fg)[3] === Flat

    # Same extent, and every atmospheric face is also a fire face.
    xa, xf = xnodes(atmosphere, Face()), xnodes(fg, Face())
    ya, yf = ynodes(atmosphere, Face()), ynodes(fg, Face())
    @test (first(xa), last(xa)) == (first(xf), last(xf))
    @test all(x -> any(≈(x), xf), xa)
    @test all(y -> any(≈(y), yf), ya)

    # Exact nesting: 8 fire cells per atmospheric cell in each direction.
    @test atmosphere.Δxᶜᵃᵃ ≈ 8 * fg.Δxᶜᵃᵃ
    @test atmosphere.Δyᵃᶜᵃ ≈ 8 * fg.Δyᵃᶜᵃ
    @test fg.Δxᶜᵃᵃ ≈ 30.0

    @test size(fire_grid(atmosphere; refinement = 1)) == (8, 6, 1)
    @test_throws ArgumentError fire_grid(atmosphere; refinement = 0)

    stretched = RectilinearGrid(size = (8, 6, 4), halo = (3, 3, 3),
                                x = collect(range(-960.0, 960.0, length = 9)) .^ 3 ./ 960^2,
                                y = (-720.0, 720.0), z = (0.0, 400.0),
                                topology = (Bounded, Bounded, Bounded))
    @test_throws ArgumentError fire_grid(stretched)
end

@testset "FireModel" begin
    atmosphere = RectilinearGrid(size = (8, 6, 4), halo = (3, 3, 3),
                                 x = (-960.0, 960.0), y = (-720.0, 720.0), z = (0.0, 400.0),
                                 topology = (Bounded, Bounded, Bounded))
    model = FireModel(fire_grid(atmosphere; refinement = 8))

    @test size(interior(model.φ)) == (64, 48, 1)
    @test size(interior(model.t_ignition)) == (64, 48, 1)
    @test all(isinf, interior(model.φ))            # no fire anywhere
    @test all(isinf, interior(model.t_ignition))
    @test model.clock.time == 0
    @test model.clock.iteration == 0

    @test keys(Oceananigans.fields(model)) == (:φ, :t_ignition)
    @test occursin("FireModel", sprint(show, model))

    # A 3D grid is not a fire grid.
    @test_throws ArgumentError FireModel(atmosphere)

    # Fire state and terrain share a field type, so the DEM sets onto the fire
    # grid as well as the atmospheric one.
    dem = Raster(joinpath(dirname(@__DIR__), "docs", "data", "marshall", "elevation.tif"))
    crs = ProjectedCRS(-105.182, 39.9575)
    h = Field{Center, Center, Nothing}(model.grid)
    set!(h, raster_topography(dem, crs))
    @test size(interior(h)) == size(interior(model.φ))
    @test !any(isnan, interior(h))
end

@testset "ignite!" begin
    dx = 10.0
    Nx = Ny = 64
    grid = RectilinearGrid(size = (Nx, Ny), halo = (3, 3),
                           x = (-Nx * dx / 2, Nx * dx / 2), y = (-Ny * dx / 2, Ny * dx / 2),
                           topology = (Bounded, Bounded, Flat))
    model = FireModel(grid)
    R = 100.0
    ignite!(model, 0.0, 0.0; radius = R)

    xs = xnodes(grid, Center())
    ys = ynodes(grid, Center())
    φ = interior(model.φ)[:, :, 1]
    t = interior(model.t_ignition)[:, :, 1]
    r = [hypot(x, y) for x in xs, y in ys]

    @test !any(isinf, φ)                             # φ is set everywhere
    @test all(φ[r .< R] .< 0)                        # burning inside
    @test all(φ[r .> R] .> 0)                        # unburned outside
    @test φ ≈ r .- R                                 # exact signed distance

    # |∇φ| = 1 away from the center singularity, so no reinitialization needed.
    ∇x = (φ[3:end, 2:end-1] .- φ[1:end-2, 2:end-1]) ./ (2dx)
    ∇y = (φ[2:end-1, 3:end] .- φ[2:end-1, 1:end-2]) ./ (2dx)
    far = r[2:end-1, 2:end-1] .> 4dx
    @test all(isapprox.(hypot.(∇x[far], ∇y[far]), 1; atol = 0.02))

    @test all(t[r .< R] .== 0)                       # ignited at clock time
    @test all(isinf, t[r .> R])                      # never ignited

    @test_throws ArgumentError ignite!(model, 0.0, 0.0; radius = 0.0)
end

@testset "ignite! composes" begin
    dx = 10.0
    Nx = Ny = 64
    grid = RectilinearGrid(size = (Nx, Ny), halo = (3, 3),
                           x = (-Nx * dx / 2, Nx * dx / 2), y = (-Ny * dx / 2, Ny * dx / 2),
                           topology = (Bounded, Bounded, Flat))
    xs, ys = xnodes(grid, Center()), ynodes(grid, Center())

    both = FireModel(grid)
    ignite!(both, -150.0, 0.0; radius = 50.0, time = 10.0)
    ignite!(both, 150.0, 0.0; radius = 80.0, time = 20.0)

    d1 = [hypot(x + 150, y) - 50 for x in xs, y in ys]
    d2 = [hypot(x - 150, y) - 80 for x in xs, y in ys]

    # Signed distance to the union, and the earliest ignition time per cell.
    @test interior(both.φ)[:, :, 1] ≈ min.(d1, d2)
    t = interior(both.t_ignition)[:, :, 1]
    @test all(t[d1 .< 0] .== 10.0)
    @test all(t[(d2 .< 0) .& (d1 .>= 0)] .== 20.0)
    @test all(isinf, t[(d1 .>= 0) .& (d2 .>= 0)])

    # Order does not matter.
    reversed = FireModel(grid)
    ignite!(reversed, 150.0, 0.0; radius = 80.0, time = 20.0)
    ignite!(reversed, -150.0, 0.0; radius = 50.0, time = 10.0)
    @test interior(reversed.φ) ≈ interior(both.φ)
    @test interior(reversed.t_ignition) ≈ interior(both.t_ignition)
end

@testset "advance!" begin
    dx = 10.0
    Nx = Ny = 64
    grid = RectilinearGrid(size = (Nx, Ny), halo = (3, 3),
                           x = (-Nx * dx / 2, Nx * dx / 2), y = (-Ny * dx / 2, Ny * dx / 2),
                           topology = (Bounded, Bounded, Flat))
    xs, ys = xnodes(grid, Center()), ynodes(grid, Center())
    r = [hypot(x, y) for x in xs, y in ys]
    R, F, Δt, n = 50.0, 2.0, 1.0, 75

    # Uniform speed: the exact solution is the distance function translated by
    # F t, except within F t of the centre, where Godunov holds the local
    # minimum fixed and the error spreads outward at speed F.
    model = FireModel(grid)
    ignite!(model, 0.0, 0.0; radius = R)
    for _ in 1:n
        advance!(model, Δt; speed = F)
    end
    @test model.clock.time == n * Δt
    @test model.clock.iteration == n

    φ = interior(model.φ)[:, :, 1]
    t = interior(model.t_ignition)[:, :, 1]
    exact = r .- R .- F * n * Δt
    far = r .> F * n * Δt + 3dx
    @test maximum(abs.(φ[far] .- exact[far])) < dx
    @test all(φ[far] .>= exact[far] .- 1e-9)        # first-order upwind lags, never leads
    @test 0.9 < count(<(0), φ) * dx^2 / (π * (R + F * n * Δt)^2) < 1.0

    # Ignition time is the arrival time (r - R) / F, to within a cell's transit.
    ring = (r .> R + 2dx) .& (r .< R + F * n * Δt - 2dx)
    @test maximum(abs.(t[ring] .- (r[ring] .- R) ./ F)) < dx / F
    @test all(t[r .< R] .== 0)                       # original ignition untouched
    @test all(isinf, t[r .> R + F * n * Δt + 2dx])   # not yet reached

    # Zero speed: nothing moves, the clock still ticks.
    still = FireModel(grid)
    ignite!(still, 0.0, 0.0; radius = R)
    φ₀ = copy(interior(still.φ))
    advance!(still, Δt; speed = 0.0)
    @test interior(still.φ) == φ₀
    @test still.clock.time == Δt

    # Speed field: no spread where speed is zero.
    half = FireModel(grid)
    ignite!(half, 0.0, 0.0; radius = R)
    φ₀ = copy(interior(half.φ))
    speed = Field{Center, Center, Nothing}(grid)
    set!(speed, (x, y) -> x > 0 ? F : 0.0)
    for _ in 1:n
        advance!(half, Δt; speed)
    end
    left = [x < 0 for x in xs, _ in ys]
    @test interior(half.φ)[:, :, 1][left] == φ₀[:, :, 1][left]
    @test all(interior(half.φ)[:, :, 1][.!left .& (r .< R + F * n * Δt - 2dx)] .< 0)

    # Float32 grid: no Float64 promotion in the kernels.
    grid32 = RectilinearGrid(Float32; size = (Nx, Ny), halo = (3, 3),
                             x = (-Nx * dx / 2, Nx * dx / 2), y = (-Ny * dx / 2, Ny * dx / 2),
                             topology = (Bounded, Bounded, Flat))
    m32 = FireModel(grid32)
    ignite!(m32, 0.0, 0.0; radius = R)
    advance!(m32, Δt; speed = F)
    @test eltype(interior(m32.φ)) == Float32
    @test interior(m32.φ) ≈ interior(FireModel(grid) |> m -> (ignite!(m, 0.0, 0.0; radius = R); advance!(m, Δt; speed = F)).φ) rtol = 1e-5
end
