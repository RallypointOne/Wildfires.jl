@testset "Boundary Conditions" begin
    @testset "ZeroNeumann default" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        prop = LevelSetPropagation()
        for _ in 1:10
            advance!(grid, F, 0.5, prop)
        end
        reinitialize!(grid, IterativeReinit(), ZeroNeumann())
        @test burn_area(grid) > 0
    end

    @testset "Dirichlet preserves edges" begin
        grid = levelset_grid(30, 30; dx=30.0)
        prop = LevelSetPropagation(bc=Dirichlet())
        ignite!(grid, 450.0, 450.0, 50.0)
        top = copy(grid.state[1, :])
        bot = copy(grid.state[end, :])
        left = copy(grid.state[:, 1])
        right = copy(grid.state[:, end])
        F = fill(10.0, size(grid))
        for _ in 1:10
            advance!(grid, F, 0.5, prop)
        end
        reinitialize!(grid, IterativeReinit(), Dirichlet())
        @test grid.state[1, :] == top
        @test grid.state[end, :] == bot
        @test grid.state[:, 1] == left
        @test grid.state[:, end] == right
    end

    @testset "Periodic runs without error" begin
        grid = levelset_grid(30, 30; dx=30.0)
        prop = LevelSetPropagation(bc=Periodic())
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        for _ in 1:10
            advance!(grid, F, 0.5, prop)
        end
        reinitialize!(grid, IterativeReinit(), Periodic())
        @test burn_area(grid) > 0
    end
end

@testset "Burnable" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "default all burnable" begin
        grid = levelset_grid(20, 20; dx=30.0)
        @test all(isinf, grid.layers.t_ignite)
        @test all(burnable(grid))
    end

    @testset "set_unburnable! marks correct cells" begin
        grid = levelset_grid(50, 50; dx=30.0)
        cx, cy, r = 750.0, 750.0, 100.0
        set_unburnable!(grid, cx, cy, r)
        xs = xcoords(grid)
        ys = ycoords(grid)
        for j in eachindex(xs), i in eachindex(ys)
            inside = hypot(xs[j] - cx, ys[i] - cy) <= r
            @test isnan(grid.layers.t_ignite[i, j]) == inside
            @test burnable(grid)[i, j] == !inside
        end
    end

    @testset "spread_rate_field! returns zero in unburnable cells" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        set_unburnable!(grid, 300.0, 300.0, 80.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        F = similar(grid.state)
        spread_rate_field!(F, model, grid)
        for j in axes(F, 2), i in axes(F, 1)
            if isnan(grid.layers.t_ignite[i, j])
                @test F[i, j] == 0.0
            end
        end
    end

    @testset "fire stops at unburnable strip" begin
        grid = levelset_grid(80, 80; dx=20.0)
        ignite!(grid, 800.0, 800.0, 50.0)
        xs = xcoords(grid)
        for j in eachindex(xs)
            if 980.0 <= xs[j] <= 1020.0
                grid.layers.t_ignite[:, j] .= NaN
            end
        end
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid, model, LevelSetPropagation(); steps=200, dt=0.3)
        for j in eachindex(xs)
            if xs[j] > 1040.0
                @test all(>(0), grid.state[:, j])
            end
        end
    end

    @testset "fire cannot spread into unburnable region" begin
        grid = levelset_grid(80, 80; dx=20.0)
        ignite!(grid, 800.0, 800.0, 50.0)
        set_unburnable!(grid, 1100.0, 800.0, 100.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid, model, LevelSetPropagation(); steps=200, dt=0.3)
        for j in axes(grid.state, 2), i in axes(grid.state, 1)
            if isnan(grid.layers.t_ignite[i, j])
                @test grid.state[i, j] >= 0
            end
        end
    end
end

@testset "Ignition Tracking" begin
    @testset "ignite! records t_ignite at current time" begin
        grid = levelset_grid(20, 20; dx=30.0)
        ignite!(grid, 300.0, 300.0, 50.0)
        for j in axes(grid.state, 2), i in axes(grid.state, 1)
            if grid.state[i, j] < 0
                @test grid.layers.t_ignite[i, j] == 0.0
            else
                @test isinf(grid.layers.t_ignite[i, j])
            end
        end
    end

    @testset "advance! records ignition time for newly burned cells" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        initial_ignited = count(isfinite, grid.layers.t_ignite)
        prop = LevelSetPropagation()
        for _ in 1:5
            advance!(grid, F, 0.5, prop)
        end
        @test count(isfinite, grid.layers.t_ignite) > initial_ignited
        for j in axes(grid.layers.t_ignite, 2), i in axes(grid.layers.t_ignite, 1)
            t_ig = grid.layers.t_ignite[i, j]
            if isfinite(t_ig) && t_ig > 0
                @test t_ig <= grid.t
            end
        end
    end
end

@testset "Solvers" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "Godunov default" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        prop = LevelSetPropagation()
        for _ in 1:10
            advance!(grid, F, 0.5, prop)
        end
        @test burn_area(grid) > 0
        @test grid.t ≈ 5.0
    end

    @testset "Superbee runs and produces burned area" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        prop = LevelSetPropagation(solver=Superbee())
        for _ in 1:10
            advance!(grid, F, 0.5, prop)
        end
        @test burn_area(grid) > 0
        @test grid.t ≈ 5.0
    end

    @testset "Superbee phi clamping" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        prop = LevelSetPropagation(solver=Superbee(phi_clamp=100.0))
        for _ in 1:20
            advance!(grid, F, 0.5, prop)
        end
        @test all(x -> -100.0 <= x <= 100.0, grid.state)
    end

    @testset "Superbee records ignition times" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        initial_ignited = count(isfinite, grid.layers.t_ignite)
        prop = LevelSetPropagation(solver=Superbee())
        for _ in 1:10
            advance!(grid, F, 0.5, prop)
        end
        @test count(isfinite, grid.layers.t_ignite) > initial_ignited
    end

    @testset "simulate! with Godunov()" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        simulate!(grid, model, LevelSetPropagation(); steps=20, dt=0.5)
        @test burn_area(grid) > 0
    end

    @testset "simulate! with Superbee()" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        simulate!(grid, model, LevelSetPropagation(solver=Superbee()); steps=20, dt=0.5)
        @test burn_area(grid) > 0
    end

    @testset "both solvers converge to similar fire areas" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_god = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_god, 750.0, 750.0, 50.0)
        simulate!(grid_god, model, LevelSetPropagation(); steps=50, dt=0.5)

        grid_sup = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_sup, 750.0, 750.0, 50.0)
        simulate!(grid_sup, model, LevelSetPropagation(solver=Superbee()); steps=50, dt=0.5)

        area_god = burn_area(grid_god)
        area_sup = burn_area(grid_sup)
        @test area_god > 0
        @test area_sup > 0
        ratio = area_sup / area_god
        @test 0.3 < ratio < 3.0
    end

    @testset "Superbee with all boundary conditions" begin
        for bc in (ZeroNeumann(), Dirichlet(), Periodic())
            grid = levelset_grid(30, 30; dx=30.0)
            ignite!(grid, 450.0, 450.0, 50.0)
            F = fill(10.0, size(grid))
            prop = LevelSetPropagation(solver=Superbee(), bc=bc)
            for _ in 1:5
                advance!(grid, F, 0.5, prop)
            end
            @test burn_area(grid) > 0
        end
    end
end

@testset "WENO5 Solver" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "WENO5 runs and produces burned area" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        prop = LevelSetPropagation(solver=WENO5())
        for _ in 1:10
            advance!(grid, F, 0.5, prop)
        end
        @test burn_area(grid) > 0
        @test grid.t ≈ 5.0
    end

    @testset "WENO5 phi clamping" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        prop = LevelSetPropagation(solver=WENO5(phi_clamp=100.0))
        for _ in 1:20
            advance!(grid, F, 0.5, prop)
        end
        @test all(x -> -100.0 <= x <= 100.0, grid.state)
    end

    @testset "WENO5 records ignition times" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        initial_ignited = count(isfinite, grid.layers.t_ignite)
        prop = LevelSetPropagation(solver=WENO5())
        for _ in 1:10
            advance!(grid, F, 0.5, prop)
        end
        @test count(isfinite, grid.layers.t_ignite) > initial_ignited
    end

    @testset "simulate! with WENO5()" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        simulate!(grid, model, LevelSetPropagation(solver=WENO5()); steps=20, dt=0.5)
        @test burn_area(grid) > 0
    end

    @testset "WENO5 with all boundary conditions" begin
        for bc in (ZeroNeumann(), Dirichlet(), Periodic())
            grid = levelset_grid(30, 30; dx=30.0)
            ignite!(grid, 450.0, 450.0, 50.0)
            F = fill(10.0, size(grid))
            prop = LevelSetPropagation(solver=WENO5(), bc=bc)
            for _ in 1:5
                advance!(grid, F, 0.5, prop)
            end
            @test burn_area(grid) > 0
        end
    end

    @testset "all three solvers converge to similar fire areas" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        areas = Float64[]
        for solver in (Godunov(), Superbee(), WENO5())
            g = levelset_grid(50, 50; dx=30.0)
            ignite!(g, 750.0, 750.0, 50.0)
            simulate!(g, model, LevelSetPropagation(solver=solver); steps=50, dt=0.5)
            push!(areas, burn_area(g))
        end
        for a in areas
            @test a > 0
            @test 0.3 < a / areas[1] < 3.0
        end
    end
end

@testset "Reinitialization" begin
    @testset "IterativeReinit runs" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        prop = LevelSetPropagation()
        for _ in 1:10
            advance!(grid, F, 0.5, prop)
        end
        reinitialize!(grid, IterativeReinit(), ZeroNeumann())
        @test burn_area(grid) > 0
    end
end

@testset "Curvature Regularization" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "curvature=0 matches default" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, LevelSetPropagation(); steps=20, dt=0.5)

        grid2 = levelset_grid(30, 30; dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, LevelSetPropagation(curvature=0.0); steps=20, dt=0.5)

        @test grid1.state == grid2.state
    end

    @testset "curvature > 0 runs and produces burned area" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        simulate!(grid, model, LevelSetPropagation(curvature=1.0); steps=20, dt=0.5)
        @test burn_area(grid) > 0
    end

    @testset "curvature produces different fire shape" begin
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, LevelSetPropagation(curvature=0.0); steps=50, dt=0.5)

        grid_curv = levelset_grid(50, 50; dx=30.0)
        ignite!(grid_curv, 750.0, 750.0, 50.0)
        simulate!(grid_curv, model, LevelSetPropagation(curvature=5.0); steps=50, dt=0.5)

        @test burn_area(grid_no) > 0
        @test burn_area(grid_curv) > 0
        @test burn_area(grid_no) != burn_area(grid_curv)
    end

    @testset "cfl_dt with curvature constraint" begin
        grid = levelset_grid(100, 100; dx=30.0)
        F = fill(10.0, size(grid))
        dt_no_curv = cfl_dt(grid, F)
        dt_curv = cfl_dt(grid, F; curvature=1.0)
        @test dt_curv <= dt_no_curv
    end
end

@testset "CFL" begin
    @testset "cfl_dt computation" begin
        grid = levelset_grid(100, 100; dx=30.0)
        F = fill(10.0, size(grid))
        @test cfl_dt(grid, F) == 0.5 * 30.0 / 10.0
        @test cfl_dt(grid, F; cfl=1.0) == 30.0 / 10.0
    end

    @testset "cfl_dt with zero spread" begin
        grid = levelset_grid(10, 10; dx=30.0)
        F = fill(0.0, size(grid))
        @test cfl_dt(grid, F) == Inf
    end

    @testset "simulate! with auto CFL" begin
        grid = levelset_grid(100, 100; dx=30.0)
        ignite!(grid, 1500.0, 1500.0, 50.0)
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid, model, LevelSetPropagation(); steps=20)
        @test burn_area(grid) > 0
        @test grid.t > 0
    end

    @testset "simulate! with explicit dt" begin
        grid = levelset_grid(30, 30; dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid, model, LevelSetPropagation(); steps=10, dt=0.5)
        @test grid.t ≈ 5.0
    end
end
