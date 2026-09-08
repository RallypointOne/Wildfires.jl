const GI = Wildfires.GI

square(x, y, s) = GI.Polygon([[(x, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y)]])

@testset "polygon_coverage" begin
    x₀, y₀ = 500_000.0, 4.4e6
    crs = ProjectedCRS(32613, (x₀, y₀))
    grid = RectilinearGrid(size = (40, 40), x = (0.0, 400.0), y = (0.0, 400.0),
                           topology = (Bounded, Bounded, Flat))
    cover = polygon_coverage([square(x₀ + 100, y₀ + 100, 20), square(x₀ + 205, y₀ + 105, 10)],
                             crs, grid; source_crs = "EPSG:32613")
    c = interior(cover)[:, :, 1]
    @test c[11, 11] == c[12, 11] == c[11, 12] == c[12, 12] == 1
    @test c[21, 11] == 0.25
    @test sum(c) ≈ 5
    @test all(0 .<= c .<= 1)
    # Overlapping polygons saturate at 1.
    both = polygon_coverage([square(x₀ + 100, y₀ + 100, 20), square(x₀ + 100, y₀ + 100, 20)],
                            crs, grid; source_crs = "EPSG:32613")
    @test maximum(interior(both)) == 1
end

@testset "ignite! from polygons" begin
    x₀, y₀ = 500_000.0, 4.4e6
    crs = ProjectedCRS(32613, (x₀, y₀))
    grid = RectilinearGrid(size = (40, 40), x = (0.0, 400.0), y = (0.0, 400.0),
                           topology = (Bounded, Bounded, Flat))
    model = FireModel(grid)
    ignite!(model, [square(x₀ + 100, y₀ + 100, 40)], crs; source_crs = "EPSG:32613", time = 30.0)
    φ = interior(model.φ)[:, :, 1]
    t = interior(model.t_ignition)[:, :, 1]
    at(x, y) = (findfirst(≈(x), xnodes(grid, Center())), findfirst(≈(y), ynodes(grid, Center())))
    @test φ[at(115, 115)...] ≈ -15          # centre of the square: 20 m to each edge, cell centre offset 5
    @test φ[at(165, 115)...] ≈ 25           # 25 m east of the edge at x = 140
    @test φ[at(5, 5)...] ≈ hypot(95, 95)    # nearest point is the corner
    @test t[at(115, 115)...] == 30
    @test isinf(t[at(165, 115)...])
    @test count(<(0), φ) == count(isfinite, t) == 16
    @test all(isfinite, φ)

    # Composes with an earlier ignition by pointwise minimum; earlier times win.
    ignite!(model, 300.0, 300.0; radius = 20.0, time = 10.0)
    ignite!(model, [square(x₀ + 280, y₀ + 280, 40)], crs; source_crs = "EPSG:32613", time = 60.0)
    @test interior(model.t_ignition)[at(305, 305)..., 1] == 10
    @test interior(model.t_ignition)[at(285, 285)..., 1] == 60
    @test interior(model.φ)[at(115, 115)..., 1] ≈ -15

    # A feature collection in lon/lat goes through the CRS transform.
    lon, lat = to_lonlat(crs, 200.0, 200.0)
    fc = GI.FeatureCollection([GI.Feature(GI.Polygon([[(lon - 1e-4, lat - 1e-4), (lon + 1e-4, lat - 1e-4),
                                                        (lon + 1e-4, lat + 1e-4), (lon - 1e-4, lat + 1e-4),
                                                        (lon - 1e-4, lat - 1e-4)]]); properties = (;))])
    fresh = FireModel(grid)
    ignite!(fresh, fc, crs)
    @test interior(fresh.φ)[at(205, 205)..., 1] < 0
    @test interior(fresh.φ)[at(5, 5)..., 1] > 200
end
