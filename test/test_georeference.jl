@testset "UTM utilities" begin
    @test utm_zone(-105.2) == 13
    @test utm_zone(151.2) == 56
    @test utm_zone(-180.0) == 1
    @test utm_zone(180.0) == 60
    @test utm_epsg(-105.2, 40.0) == 32613
    @test utm_epsg(151.2, -33.9) == 32756
end

@testset "ProjectedCRS" begin
    lon, lat = -105.182, 39.9575
    crs = ProjectedCRS(lon, lat)
    @test crs.epsg == 32613

    # The defining point sits at the grid origin.
    @test all(isapprox.(to_grid(crs, lon, lat), (0, 0); atol = 1e-6))

    # Round trip through the projection.
    lon′, lat′ = to_lonlat(crs, to_grid(crs, lon, lat)...)
    @test lon′ ≈ lon
    @test lat′ ≈ lat

    # Both constructors are reachable; the explicit form round-trips.
    @test ProjectedCRS(crs.epsg, crs.origin) == crs
    @test ProjectedCRS(32613, (0, 0)).origin === (0.0, 0.0)

    # 1 km east of the origin is ~1 km of easting.
    x, y = to_grid(crs, to_lonlat(crs, 1000.0, 0.0)...)
    @test x ≈ 1000.0 atol = 1e-6
    @test y ≈ 0.0 atol = 1e-6
end

@testset "raster_topography" begin
    dem_path = joinpath(dirname(@__DIR__), "docs", "data", "marshall", "elevation.tif")
    dem = Raster(dem_path)
    lo, hi = extrema(filter(!isnan, vec(parent(read(dem)))))

    crs = ProjectedCRS(-105.182, 39.9575)
    h = raster_topography(dem, crs)
    @test lo <= h(0.0, 0.0) <= hi

    # The closure is the `topography` argument of Breeze's terrain-following
    # coordinate: at the bottom face, physical altitude is the terrain height.
    Nx, Ny, Nz, dx = 64, 48, 8, 100.0
    z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, 12e3, length = Nz + 1));
                                                     formulation = LinearDecay())
    grid = RectilinearGrid(size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (-Nx * dx / 2, Nx * dx / 2),
                           y = (-Ny * dx / 2, Ny * dx / 2),
                           z = z_faces,
                           topology = (Bounded, Bounded, Bounded))
    materialize_terrain!(grid, h)
    z_bottom = [znode(i, j, 1, grid, Center(), Center(), Face()) for i in 1:Nx, j in 1:Ny]

    @test !any(isnan, z_bottom)
    @test all(lo .<= z_bottom .<= hi)          # bilinear stays within the DEM range
    @test z_bottom ≈ [h(x, y) for x in xnodes(grid, Center()), y in ynodes(grid, Center())]

    # Points off the raster get fill_value.
    far = ProjectedCRS(0.0, 0.0)
    @test raster_topography(dem, far; fill_value = -1)(0.0, 0.0) == -1
end

@testset "raster_sampler" begin
    # Synthetic raster already in UTM 13N, so the warp is the identity and
    # sampled values can be checked against the array. GDAL-style lookups hold
    # cell starts; the sampler must place values at cell centres.
    Lookups = Rasters.DimensionalData.Lookups
    Δ = 30.0
    xs = 500_000.0 .+ (0:9) .* Δ
    ys = 4.4e6 .+ (0:7) .* Δ
    A = [10.0i + j for i in 1:10, j in 1:8]
    raster = Raster(A, (X(xs; sampling = Lookups.Intervals(Lookups.Start())),
                        Y(ys; sampling = Lookups.Intervals(Lookups.Start())));
                    crs = Rasters.EPSG(32613))
    crs = ProjectedCRS(32613, (first(xs) + Δ / 2, first(ys) + Δ / 2))   # (0, 0) at the centre of A[1, 1]

    near = raster_sampler(raster, crs; method = :near, fill_value = -1)
    @test near(0.0, 0.0) == A[1, 1]
    @test near(Δ, 0.0) == A[2, 1]
    @test near(0.0, Δ) == A[1, 2]
    @test near(0.4Δ, 0.0) == A[1, 1]
    @test near(0.6Δ, 0.0) == A[2, 1]
    @test near(9Δ, 7Δ) == A[10, 8]
    @test near(-0.6Δ, 0.0) == -1
    @test near(9.6Δ, 0.0) == -1

    lin = raster_sampler(raster, crs; method = :bilinear, fill_value = -1)
    @test lin(0.0, 0.0) ≈ A[1, 1]
    @test lin(Δ, Δ) ≈ A[2, 2]
    @test lin(Δ / 2, 0.0) ≈ (A[1, 1] + A[2, 1]) / 2
    @test lin(Δ / 2, Δ / 2) ≈ (A[1, 1] + A[2, 1] + A[1, 2] + A[2, 2]) / 4
    @test lin(-0.4Δ, 0.0) ≈ A[1, 1]           # edge strip clamps
    @test lin(-0.6Δ, 0.0) == -1
    @test raster_topography(raster, crs)(Δ, 0.0) ≈ A[2, 1]

    @test_throws ArgumentError raster_sampler(raster, crs; method = :cubic)

    # Categorical LANDFIRE fuel codes survive the warp and the lookup.
    fuel = Raster(joinpath(dirname(@__DIR__), "docs", "data", "marshall", "fuel.tif"))
    codes = Set(filter(!isnan, vec(Float64.(parent(read(fuel))))))
    code = raster_sampler(fuel, ProjectedCRS(-105.182, 39.9575); method = :near)
    grid = RectilinearGrid(size = (64, 48), x = (-3200, 3200), y = (-2400, 2400),
                           topology = (Bounded, Bounded, Flat))
    field = Field{Center, Center, Nothing}(grid)
    set!(field, code)
    @test Set(interior(field)) ⊆ codes
    @test length(Set(interior(field))) > 1
end

@testset "raster_time_series" begin
    Lookups = Rasters.DimensionalData.Lookups
    Δ = 30.0
    xs = 500_000.0 .+ (0:9) .* Δ
    ys = 4.4e6 .+ (0:7) .* Δ
    make(v) = Raster(fill(v, 10, 8), (X(xs; sampling = Lookups.Intervals(Lookups.Start())),
                                      Y(ys; sampling = Lookups.Intervals(Lookups.Start())));
                     crs = Rasters.EPSG(32613))
    crs = ProjectedCRS(32613, (first(xs) + Δ / 2, first(ys) + Δ / 2))
    grid = RectilinearGrid(size = (8, 6), x = (0.0, 240.0), y = (0.0, 180.0),
                           topology = (Bounded, Bounded, Flat))
    series = raster_time_series([make(1.0), make(3.0)], [0.0, 10.0], grid, crs)
    @test series.times == [0.0, 10.0]
    @test all(interior(series[1]) .≈ 1) && all(interior(series[2]) .≈ 3)
    @test all(interior(series[Time(5.0)]) .≈ 2)
    @test series[3, 2, 1, Time(2.5)] ≈ 1.5
    @test all(interior(series[Time(50.0)]) .≈ 3)            # Clamp
    @test_throws ArgumentError raster_time_series([make(1.0)], [0.0, 10.0], grid, crs)
end
