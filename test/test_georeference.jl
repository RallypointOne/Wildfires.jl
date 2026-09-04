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
