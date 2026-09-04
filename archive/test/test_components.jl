@testset "NFFL lookup" begin
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
end

@testset "Elliptical Fire Spread" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "length_to_breadth" begin
        @test length_to_breadth(0.0) ≈ 1.0 atol=0.01
        @test length_to_breadth(2.0) > 1.0
        @test length_to_breadth(5.0) > length_to_breadth(2.0)
        @test length_to_breadth(0.0; formula=:green) == 1.0
        @test length_to_breadth(2.0; formula=:green) > 1.0
    end

    @testset "fire_eccentricity" begin
        @test fire_eccentricity(1.0) ≈ 0.0
        @test fire_eccentricity(3.0) ≈ sqrt(8.0) / 3.0 atol=1e-10
        @test 0.0 < fire_eccentricity(2.0) < 1.0
    end

    @testset "CosineShape default" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        @test model.shape isa CosineShape
    end

    @testset "EllipticalShape construction" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain(), EllipticalShape())
        @test model.shape isa EllipticalShape
        @test model.shape.formula == :anderson
    end

    @testset "elliptical simulate! runs" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain(), EllipticalShape())
        simulate!(grid, model, LevelSetPropagation(); steps=20, dt=0.5)
        @test burn_area(grid) > 0
    end

    @testset "slope-only produces elongated fire with elliptical" begin
        grid = levelset_grid(80, 80; dx=20.0)
        ignite!(grid, 800.0, 800.0, 50.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=0.0), UniformMoisture(M),
            UniformSlope(slope=0.6, aspect=0.0), EllipticalShape())
        simulate!(grid, model, LevelSetPropagation(); steps=80, dt=0.5)
        @test burn_area(grid) > 0
        burned_cells = grid.state .< 0
        xs = [j for i in axes(burned_cells, 1) for j in axes(burned_cells, 2) if burned_cells[i, j]]
        ys = [i for i in axes(burned_cells, 1) for j in axes(burned_cells, 2) if burned_cells[i, j]]
        @test !isempty(xs)
        x_span = maximum(xs) - minimum(xs)
        y_span = maximum(ys) - minimum(ys)
        @test x_span != y_span
    end

    @testset "elliptical produces different fire shape than cosine" begin
        grid_cos = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_cos, 750.0, 750.0, 50.0)
        model_cos = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_ell = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_ell, 750.0, 750.0, 50.0)
        model_ell = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain(), EllipticalShape())

        simulate!(grid_cos, model_cos, LevelSetPropagation(); steps=30, dt=0.5)
        simulate!(grid_ell, model_ell, LevelSetPropagation(); steps=30, dt=0.5)

        @test burn_area(grid_cos) > 0
        @test burn_area(grid_ell) > 0
        @test burn_area(grid_ell) != burn_area(grid_cos)
    end
end

@testset "Burnout" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "burnout=nothing preserves current behavior" begin
        grid1 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid1, model, LevelSetPropagation(); steps=20, dt=0.5)

        grid2 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, LevelSetPropagation(); steps=20, dt=0.5, burnout=nothing)

        @test grid1.state == grid2.state
    end

    @testset "NoBurnout() matches burnout=nothing" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, LevelSetPropagation(); steps=20, dt=0.5, burnout=nothing)

        grid2 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, LevelSetPropagation(); steps=20, dt=0.5, burnout=NoBurnout())

        @test grid1.state == grid2.state
    end

    @testset "ExponentialBurnout limits fire spread" begin
        t_r = residence_time(SHORT_GRASS)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, LevelSetPropagation(); steps=200, dt=0.5)

        grid_bo = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_bo, 750.0, 750.0, 50.0)
        simulate!(grid_bo, model, LevelSetPropagation(); steps=200, dt=0.5, burnout=ExponentialBurnout(t_r))

        @test burn_area(grid_bo) <= burn_area(grid_no)
    end

    @testset "backward compat: burnout=Real coerces to ExponentialBurnout" begin
        t_r = residence_time(SHORT_GRASS)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, LevelSetPropagation(); steps=20, dt=0.5, burnout=t_r)

        grid2 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, LevelSetPropagation(); steps=20, dt=0.5, burnout=ExponentialBurnout(t_r))

        @test grid1.state == grid2.state
    end

    @testset "ExponentialBurnout produces less burn area than NoBurnout" begin
        t_r = residence_time(CHAPARRAL)
        model = FireModel(RothermelROS(CHAPARRAL), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, LevelSetPropagation(); steps=200, dt=0.5, burnout=NoBurnout())

        grid_bo = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_bo, 750.0, 750.0, 50.0)
        simulate!(grid_bo, model, LevelSetPropagation(); steps=200, dt=0.5, burnout=ExponentialBurnout(t_r))

        @test burn_area(grid_bo) <= burn_area(grid_no)
    end

    @testset "LinearBurnout limits fire spread" begin
        t_r = residence_time(CHAPARRAL)
        model = FireModel(RothermelROS(CHAPARRAL), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, LevelSetPropagation(); steps=200, dt=0.5)

        grid_bo = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_bo, 750.0, 750.0, 50.0)
        simulate!(grid_bo, model, LevelSetPropagation(); steps=200, dt=0.5, burnout=LinearBurnout(t_r))

        @test burn_area(grid_bo) <= burn_area(grid_no)
    end
end

@testset "Burn-in" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "burnin=nothing preserves current behavior" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, LevelSetPropagation(); steps=20, dt=0.5)

        grid2 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, LevelSetPropagation(); steps=20, dt=0.5, burnin=nothing)

        @test grid1.state == grid2.state
    end

    @testset "NoBurnin() matches burnin=nothing" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, LevelSetPropagation(); steps=20, dt=0.5, burnin=nothing)

        grid2 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, LevelSetPropagation(); steps=20, dt=0.5, burnin=NoBurnin())

        @test grid1.state == grid2.state
    end

    @testset "ExponentialBurnin limits fire spread" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, LevelSetPropagation(); steps=200, dt=0.5)

        grid_bi = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_bi, 750.0, 750.0, 50.0)
        simulate!(grid_bi, model, LevelSetPropagation(); steps=200, dt=0.5, burnin=ExponentialBurnin(0.5))

        @test burn_area(grid_bi) <= burn_area(grid_no)
    end

    @testset "LinearBurnin limits fire spread" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, LevelSetPropagation(); steps=200, dt=0.5)

        grid_bi = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_bi, 750.0, 750.0, 50.0)
        simulate!(grid_bi, model, LevelSetPropagation(); steps=200, dt=0.5, burnin=LinearBurnin(1.0))

        @test burn_area(grid_bi) <= burn_area(grid_no)
    end

    @testset "burnin + burnout combined" begin
        t_r = residence_time(CHAPARRAL)
        model = FireModel(RothermelROS(CHAPARRAL), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_both = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_both, 750.0, 750.0, 50.0)
        simulate!(grid_both, model, LevelSetPropagation(); steps=200, dt=0.5,
            burnout=ExponentialBurnout(t_r), burnin=ExponentialBurnin(0.5))

        grid_none = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_none, 750.0, 750.0, 50.0)
        simulate!(grid_none, model, LevelSetPropagation(); steps=200, dt=0.5)

        @test burn_area(grid_both) <= burn_area(grid_none)
    end
end
