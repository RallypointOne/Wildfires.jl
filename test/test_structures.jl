const GI = Wildfires.GI

# Axis-aligned square footprint with lower-left corner (x, y) and side `s`,
# in UTM 13N metres.
square(x, y, s) = GI.Polygon([[(x, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y)]])

@testset "BuildingClass" begin
    c = BuildingClass()
    @test !c.fire_resistant && c.peak_hrr == 25e3
    @test BuildingClass(fire_resistant = true).fire_resistant
    @test_throws ArgumentError BuildingClass(t_growth = 10.0, t_plateau = 5.0)
    @test BuildingClass{Float32}(c).peak_hrr isa Float32

    hrr = Wildfires._hrr
    @test hrr(c, -1.0) == 0
    @test hrr(c, c.t_growth / 2) ≈ c.peak_hrr / 2
    @test hrr(c, c.t_growth) ≈ c.peak_hrr
    @test hrr(c, (c.t_growth + c.t_plateau) / 2) ≈ c.peak_hrr
    @test hrr(c, (c.t_plateau + c.t_decay) / 2) ≈ c.peak_hrr / 2
    @test hrr(c, c.t_decay + 1) == 0
end

@testset "Structures" begin
    # 40 × 40 cells of 10 m; grid (0, 0) is UTM (500000, 4.4e6), and the
    # footprints are given in that CRS, so the transform is a translation.
    x₀, y₀ = 500_000.0, 4.4e6
    crs = ProjectedCRS(32613, (x₀, y₀))
    grid = RectilinearGrid(size = (40, 40), x = (0.0, 400.0), y = (0.0, 400.0),
                           topology = (Bounded, Bounded, Flat))
    classes = (BuildingClass(), BuildingClass(fire_resistant = true))
    features = GI.FeatureCollection([
        GI.Feature(square(x₀ + 100, y₀ + 100, 10); properties = (kind = "house",)),   # cell (11, 11)
        GI.Feature(square(x₀ + 130, y₀ + 100, 10); properties = (kind = "shed",)),    # cell (14, 11)
        GI.Feature(square(x₀ + 205, y₀ + 105, 10); properties = (kind = "house",)),   # quarter of 4 cells
        GI.Feature(square(x₀ + 900, y₀ + 900, 10); properties = (kind = "house",)),   # off the grid
    ])
    structures = Structures(features, crs, grid; source_crs = "EPSG:32613", classes,
                            class = f -> GI.properties(f).kind == "shed" ? 2 : 1, radius = 60.0)

    @test length(structures) == 3
    @test structures.area ≈ [100, 100, 100]
    @test structures.x ≈ [105, 135, 210] && structures.y ≈ [105, 105, 110]
    @test structures.class == [1, 2, 1]
    @test all(isinf, structures.t_ignition)

    cover = interior(structures.footprint_fraction)[:, :, 1]
    @test cover[11, 11] == 1 && cover[14, 11] == 1
    @test cover[21, 11] == cover[22, 11] == cover[21, 12] == cover[22, 12] == 0.25
    @test sum(cover) ≈ 3
    @test structures.offsets == [1, 2, 3, 7]
    @test (structures.cell_i[1], structures.cell_j[1]) == (11, 11)

    plan = interior(structures.plan_dimension)[:, :, 1]
    sep = interior(structures.separation)[:, :, 1]
    nonburnable = interior(structures.nonburnable_fraction)[:, :, 1]
    @test plan[11, 11] ≈ 10                      # both neighbours within 60 m
    @test sep[11, 11] ≈ 20                       # 30 m centroid gap minus two half-plans
    @test nonburnable[11, 11] ≈ 0.5              # the shed is fire-resistant
    @test sep[21, 11] ≈ 60 && plan[21, 11] ≈ 10  # isolated: separation capped at radius
    @test plan[1, 1] == 0 && sep[1, 1] == 0      # 141 m away: not urban

    # A multipolygon yields one structure per polygon; a bare polygon works too.
    multi = GI.MultiPolygon([square(x₀ + 10, y₀ + 10, 10), square(x₀ + 50, y₀ + 50, 10)])
    @test length(Structures([multi, square(x₀ + 300, y₀ + 300, 10)], crs, grid; source_crs = "EPSG:32613")) == 3
    @test_throws ArgumentError Structures(features, crs, grid; source_crs = "EPSG:32613", class = Returns(3))

    # Ignition times come from the level set, earliest footprint cell first.
    model = FireModel(grid)
    ignite!(model, 105.0, 105.0; radius = 12.0)
    update_structures!(structures, model)
    @test structures.t_ignition[1] == 0
    @test isinf(structures.t_ignition[2]) && isinf(structures.t_ignition[3])
    for _ in 1:30
        advance!(model, 1.0; speed = 1.0)
    end
    update_structures!(structures, model)
    @test 0 < structures.t_ignition[2] < 30
    @test isinf(structures.t_ignition[3])
    @test structures.t_ignition[1] == 0                      # never reset

    # Heat release follows the class curve times footprint area.
    c = classes[1]
    t₂ = structures.t_ignition[2]
    @test heat_release(structures, -1.0) == [0, 0, 0]
    hrr = heat_release(structures, t₂ + c.t_growth)
    @test hrr[2] ≈ 100 * c.peak_hrr
    @test hrr[3] == 0
    @test heat_release(structures, t₂ + c.t_decay + 1)[2] == 0
    out = similar(structures.area)
    @test heat_release!(out, structures, t₂ + c.t_growth) === out

    # Empty input is allowed.
    empty = Structures(GI.Polygon[], crs, grid; source_crs = "EPSG:32613")
    @test length(empty) == 0
    @test update_structures!(empty, model) === empty
    @test isempty(heat_release(empty, 0.0))
end

@testset "spread_rate! with structures" begin
    x₀, y₀ = 500_000.0, 4.4e6
    crs = ProjectedCRS(32613, (x₀, y₀))
    grid = RectilinearGrid(size = (40, 40), x = (0.0, 400.0), y = (0.0, 400.0),
                           topology = (Bounded, Bounded, Flat))
    footprints = [square(x₀ + 100, y₀ + 100, 10), square(x₀ + 130, y₀ + 100, 10)]
    structures = Structures(footprints, crs, grid; source_crs = "EPSG:32613", radius = 60.0)

    model = FireModel(grid)
    ignite!(model, 55.0, 105.0; radius = 10.0)   # cell (11, 11) sits due east: normal (1, 0)
    speed = Field{Center, Center, Nothing}(grid)
    bed = FuelBed(Wildfires.NFFL.SHORT_GRASS)
    wet = FuelClasses(0.5, 1.0)                  # above extinction: no vegetation spread
    U = 15.0

    spread_rate!(speed, model, bed, wet; wind = (U, 0.0), structures)
    s = interior(speed)[:, :, 1]
    head, flank, back = hamada_rates(HamadaModel(), U, 10.0, 20.0, 0.0)
    @test s[11, 11] ≈ head
    @test s[1, 1] == 0                           # no vegetation, no structures
    @test all(isfinite, s) && all(s .>= 0)
    @test count(>(0), s) == count(>(0), interior(structures.plan_dimension))

    # Without structures the same call gives zero everywhere; with dry grass the
    # cell takes the larger of the two rates.
    spread_rate!(speed, model, bed, wet; wind = (U, 0.0))
    @test all(interior(speed) .== 0)
    dry = FuelClasses(0.08, 1.0)
    spread_rate!(speed, model, bed, dry; wind = (U, 0.0), structures)
    grass = spread_rate(bed, dry, wind_adjustment(bed) * U, 0.0)
    @test interior(speed)[11, 11, 1] ≈ max(grass, head)
end
