@testset "AbstractFuel and NFFL lookup" begin
    @testset "NFFL_MODELS has 13 entries" begin
        @test length(NFFL_MODELS) == 13
        @test NFFL_MODELS[1] === SHORT_GRASS
        @test NFFL_MODELS[13] === HEAVY_SLASH
    end

    @testset "nffl_model lookup" begin
        @test nffl_model(1) === SHORT_GRASS
        @test nffl_model(4) === CHAPARRAL
        @test nffl_model(13) === HEAVY_SLASH
        @test nffl_model(0) === nothing
        @test nffl_model(14) === nothing
        @test nffl_model(99) === nothing
    end

    @testset "AbstractFuel with RothermelModel" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

        # Define a simple spatially varying fuel
        struct TestFuel <: AbstractFuel end
        (::TestFuel)(t, x, y) = x < 750.0 ? SHORT_GRASS : CHAPARRAL

        grid = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid, 750.0, 750.0, 50.0)
        model = RothermelModel(TestFuel(), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid, model, steps=30, dt=0.5)
        @test burn_area(grid) > 0
    end
end

@testset "Elliptical Fire Spread" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "length_to_breadth" begin
        @test length_to_breadth(0.0) ≈ 1.0 atol=0.01
        @test length_to_breadth(2.0) > 1.0
        @test length_to_breadth(5.0) > length_to_breadth(2.0)
        # Green formula
        @test length_to_breadth(0.0; formula=:green) == 1.0  # clamped to 1
        @test length_to_breadth(2.0; formula=:green) > 1.0
    end

    @testset "fire_eccentricity" begin
        @test fire_eccentricity(1.0) ≈ 0.0
        @test fire_eccentricity(3.0) ≈ sqrt(8.0) / 3.0 atol=1e-10
        @test 0.0 < fire_eccentricity(2.0) < 1.0
    end

    @testset "CosineBlending default" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        @test model.blending_mode isa CosineBlending
    end

    @testset "EllipticalBlending construction" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain(), EllipticalBlending())
        @test model.blending_mode isa EllipticalBlending
        @test model.blending_mode.formula == :anderson
    end

    @testset "elliptical simulate! runs" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain(), EllipticalBlending())
        simulate!(grid, model, steps=20, dt=0.5)
        @test burn_area(grid) > 0
    end

    @testset "slope-only produces elongated fire with elliptical" begin
        grid = LevelSetGrid(80, 80, dx=20.0)
        ignite!(grid, 800.0, 800.0, 50.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=0.0), UniformMoisture(M),
            UniformSlope(slope=0.6, aspect=0.0), EllipticalBlending())
        simulate!(grid, model, steps=80, dt=0.5)
        @test burn_area(grid) > 0
        # Fire should be asymmetric (elongated uphill) not circular
        burned = grid.φ .< 0
        xs = [j for i in axes(burned, 1) for j in axes(burned, 2) if burned[i, j]]
        ys = [i for i in axes(burned, 1) for j in axes(burned, 2) if burned[i, j]]
        @test !isempty(xs)
        x_span = maximum(xs) - minimum(xs)
        y_span = maximum(ys) - minimum(ys)
        @test x_span != y_span  # not circular
    end

    @testset "elliptical produces different fire shape than cosine" begin
        grid_cos = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_cos, 750.0, 750.0, 50.0)
        model_cos = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_ell = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_ell, 750.0, 750.0, 50.0)
        model_ell = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain(), EllipticalBlending())

        simulate!(grid_cos, model_cos, steps=30, dt=0.5)
        simulate!(grid_ell, model_ell, steps=30, dt=0.5)

        # Both burn, but produce different perimeters
        @test burn_area(grid_cos) > 0
        @test burn_area(grid_ell) > 0
        @test burn_area(grid_ell) != burn_area(grid_cos)
    end
end

@testset "Burnout" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "burnout=nothing preserves current behavior" begin
        grid1 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid1, model, steps=20, dt=0.5)

        grid2 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, steps=20, dt=0.5, burnout=nothing)

        @test grid1.φ == grid2.φ
    end

    @testset "NoBurnout() matches burnout=nothing" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, steps=20, dt=0.5, burnout=nothing)

        grid2 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, steps=20, dt=0.5, burnout=NoBurnout())

        @test grid1.φ == grid2.φ
    end

    @testset "ExponentialBurnout limits fire spread" begin
        t_r = residence_time(SHORT_GRASS)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, steps=200, dt=0.5)

        grid_bo = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_bo, 750.0, 750.0, 50.0)
        simulate!(grid_bo, model, steps=200, dt=0.5, burnout=ExponentialBurnout(t_r))

        # With burnout, fire should spread less or equal
        @test burn_area(grid_bo) <= burn_area(grid_no)
    end

    @testset "backward compat: burnout=Real coerces to ExponentialBurnout" begin
        t_r = residence_time(SHORT_GRASS)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, steps=20, dt=0.5, burnout=t_r)

        grid2 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, steps=20, dt=0.5, burnout=ExponentialBurnout(t_r))

        @test grid1.φ == grid2.φ
    end

    @testset "ExponentialBurnout produces less burn area than NoBurnout" begin
        t_r = residence_time(CHAPARRAL)
        model = RothermelModel(CHAPARRAL, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, steps=200, dt=0.5, burnout=NoBurnout())

        grid_bo = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_bo, 750.0, 750.0, 50.0)
        simulate!(grid_bo, model, steps=200, dt=0.5, burnout=ExponentialBurnout(t_r))

        @test burn_area(grid_bo) <= burn_area(grid_no)
    end

    @testset "LinearBurnout limits fire spread" begin
        t_r = residence_time(CHAPARRAL)
        model = RothermelModel(CHAPARRAL, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, steps=200, dt=0.5)

        grid_bo = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_bo, 750.0, 750.0, 50.0)
        simulate!(grid_bo, model, steps=200, dt=0.5, burnout=LinearBurnout(t_r))

        @test burn_area(grid_bo) <= burn_area(grid_no)
    end
end

@testset "Burn-in" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "burnin=nothing preserves current behavior" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, steps=20, dt=0.5)

        grid2 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, steps=20, dt=0.5, burnin=nothing)

        @test grid1.φ == grid2.φ
    end

    @testset "NoBurnin() matches burnin=nothing" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, steps=20, dt=0.5, burnin=nothing)

        grid2 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, steps=20, dt=0.5, burnin=NoBurnin())

        @test grid1.φ == grid2.φ
    end

    @testset "ExponentialBurnin limits fire spread" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, steps=200, dt=0.5)

        grid_bi = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_bi, 750.0, 750.0, 50.0)
        simulate!(grid_bi, model, steps=200, dt=0.5, burnin=ExponentialBurnin(0.5))

        @test burn_area(grid_bi) <= burn_area(grid_no)
    end

    @testset "LinearBurnin limits fire spread" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, steps=200, dt=0.5)

        grid_bi = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_bi, 750.0, 750.0, 50.0)
        simulate!(grid_bi, model, steps=200, dt=0.5, burnin=LinearBurnin(1.0))

        @test burn_area(grid_bi) <= burn_area(grid_no)
    end

    @testset "burnin + burnout combined" begin
        t_r = residence_time(CHAPARRAL)
        model = RothermelModel(CHAPARRAL, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_both = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_both, 750.0, 750.0, 50.0)
        simulate!(grid_both, model, steps=200, dt=0.5,
            burnout=ExponentialBurnout(t_r), burnin=ExponentialBurnin(0.5))

        grid_none = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_none, 750.0, 750.0, 50.0)
        simulate!(grid_none, model, steps=200, dt=0.5)

        # Combined burnin + burnout should spread less than no scaling
        @test burn_area(grid_both) <= burn_area(grid_none)
    end
end
