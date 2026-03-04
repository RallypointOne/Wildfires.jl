@testset "Boundary Conditions" begin
    @testset "ZeroNeumann default" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        @test grid.bc isa ZeroNeumann
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        for _ in 1:10
            advance!(grid, F, 0.5)
        end
        reinitialize!(grid)
        @test burn_area(grid) > 0
    end

    @testset "Dirichlet preserves edges" begin
        grid = LevelSetGrid(30, 30, dx=30.0, bc=Dirichlet())
        @test grid.bc isa Dirichlet
        ignite!(grid, 450.0, 450.0, 50.0)
        # Record edge values after ignition (before advance/reinit)
        top = copy(grid.φ[1, :])
        bot = copy(grid.φ[end, :])
        left = copy(grid.φ[:, 1])
        right = copy(grid.φ[:, end])
        F = fill(10.0, size(grid))
        for _ in 1:10
            advance!(grid, F, 0.5)
        end
        reinitialize!(grid)
        @test grid.φ[1, :] == top
        @test grid.φ[end, :] == bot
        @test grid.φ[:, 1] == left
        @test grid.φ[:, end] == right
    end

    @testset "Periodic runs without error" begin
        grid = LevelSetGrid(30, 30, dx=30.0, bc=Periodic())
        @test grid.bc isa Periodic
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        for _ in 1:10
            advance!(grid, F, 0.5)
        end
        reinitialize!(grid)
        @test burn_area(grid) > 0
    end

    @testset "Adapt roundtrip preserves BC" begin
        for bc in (ZeroNeumann(), Dirichlet(), Periodic())
            grid = LevelSetGrid(20, 20, dx=30.0, bc=bc)
            grid2 = Adapt.adapt(Array, grid)
            @test typeof(grid2.bc) == typeof(bc)
        end
    end

    @testset "show includes non-default BC" begin
        grid = LevelSetGrid(10, 10, dx=30.0, bc=Periodic())
        s = sprint(show, MIME("text/plain"), grid)
        @test contains(s, "Periodic")

        grid2 = LevelSetGrid(10, 10, dx=30.0)
        s2 = sprint(show, MIME("text/plain"), grid2)
        @test !contains(s2, "Neumann")
    end
end

@testset "Burnable" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "default all burnable" begin
        grid = LevelSetGrid(20, 20, dx=30.0)
        @test all(isinf, grid.t_ignite)
        @test all(burnable(grid))
    end

    @testset "set_unburnable! marks correct cells" begin
        grid = LevelSetGrid(50, 50, dx=30.0)
        cx, cy, r = 750.0, 750.0, 100.0
        set_unburnable!(grid, cx, cy, r)
        xs = xcoords(grid)
        ys = ycoords(grid)
        for j in eachindex(xs), i in eachindex(ys)
            inside = hypot(xs[j] - cx, ys[i] - cy) <= r
            @test isnan(grid.t_ignite[i, j]) == inside
            @test burnable(grid)[i, j] == !inside
        end
    end

    @testset "spread_rate_field! returns zero in unburnable cells" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        set_unburnable!(grid, 300.0, 300.0, 80.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        F = similar(grid.φ)
        spread_rate_field!(F, model, grid)
        for j in axes(F, 2), i in axes(F, 1)
            if isnan(grid.t_ignite[i, j])
                @test F[i, j] == 0.0
            end
        end
    end

    @testset "spread_rate_field! generic model respects burnable" begin
        grid = LevelSetGrid(20, 20, dx=30.0)
        set_unburnable!(grid, 300.0, 300.0, 80.0)
        const_model = (t, x, y) -> 5.0
        F = similar(grid.φ)
        spread_rate_field!(F, const_model, grid)
        for j in axes(F, 2), i in axes(F, 1)
            if isnan(grid.t_ignite[i, j])
                @test F[i, j] == 0.0
            else
                @test F[i, j] == 5.0
            end
        end
    end

    @testset "fire stops at unburnable strip" begin
        grid = LevelSetGrid(80, 80, dx=20.0)
        ignite!(grid, 800.0, 800.0, 50.0)
        # Vertical unburnable strip at x ≈ 1000m
        xs = xcoords(grid)
        for j in eachindex(xs)
            if 980.0 <= xs[j] <= 1020.0
                grid.t_ignite[:, j] .= NaN
            end
        end
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid, model, steps=200, dt=0.3)
        # Fire should not cross the strip: cells beyond x=1020 should be unburned
        for j in eachindex(xs)
            if xs[j] > 1040.0
                @test all(>(0), grid.φ[:, j])
            end
        end
    end

    @testset "fire cannot spread into unburnable region" begin
        grid = LevelSetGrid(80, 80, dx=20.0)
        ignite!(grid, 800.0, 800.0, 50.0)
        # Place unburnable circle ahead of the fire
        set_unburnable!(grid, 1100.0, 800.0, 100.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid, model, steps=200, dt=0.3)
        # No unburnable cell should have φ < 0
        for j in axes(grid.φ, 2), i in axes(grid.φ, 1)
            if isnan(grid.t_ignite[i, j])
                @test grid.φ[i, j] >= 0
            end
        end
    end

    @testset "show includes unburnable count" begin
        grid = LevelSetGrid(20, 20, dx=30.0)
        set_unburnable!(grid, 300.0, 300.0, 50.0)
        s = sprint(show, MIME("text/plain"), grid)
        @test contains(s, "unburnable=")
    end

    @testset "show omits unburnable when all burnable" begin
        grid = LevelSetGrid(10, 10, dx=30.0)
        s = sprint(show, MIME("text/plain"), grid)
        @test !contains(s, "unburnable")
    end
end

@testset "Ignition Tracking" begin
    @testset "ignite! records t_ignite at current time" begin
        grid = LevelSetGrid(20, 20, dx=30.0)
        ignite!(grid, 300.0, 300.0, 50.0)
        # Cells inside the ignition circle should have t_ignite = 0.0
        for j in axes(grid.φ, 2), i in axes(grid.φ, 1)
            if grid.φ[i, j] < 0
                @test grid.t_ignite[i, j] == 0.0
            else
                @test isinf(grid.t_ignite[i, j])
            end
        end
    end

    @testset "advance! records ignition time for newly burned cells" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        initial_ignited = count(isfinite, grid.t_ignite)
        for _ in 1:5
            advance!(grid, F, 0.5)
        end
        # More cells should be ignited now
        @test count(isfinite, grid.t_ignite) > initial_ignited
        # Newly ignited cells should have t_ignite > 0
        for j in axes(grid.t_ignite, 2), i in axes(grid.t_ignite, 1)
            t_ig = grid.t_ignite[i, j]
            if isfinite(t_ig) && t_ig > 0
                @test t_ig <= grid.t
            end
        end
    end

    @testset "show includes ignited count" begin
        grid = LevelSetGrid(20, 20, dx=30.0)
        ignite!(grid, 300.0, 300.0, 50.0)
        s = sprint(show, MIME("text/plain"), grid)
        @test contains(s, "ignited=")
    end
end

@testset "Solvers" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "Godunov() matches default advance!" begin
        grid1 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid1))
        for _ in 1:10
            advance!(grid1, F, 0.5)
        end

        grid2 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        for _ in 1:10
            advance!(grid2, F, 0.5, Godunov())
        end

        @test grid1.φ == grid2.φ
        @test grid1.t == grid2.t
    end

    @testset "Superbee runs and produces burned area" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        for _ in 1:10
            advance!(grid, F, 0.5, Superbee())
        end
        @test burn_area(grid) > 0
        @test grid.t ≈ 5.0
    end

    @testset "Superbee phi clamping" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        solver = Superbee(phi_clamp=100.0)
        for _ in 1:20
            advance!(grid, F, 0.5, solver)
        end
        @test all(x -> -100.0 <= x <= 100.0, grid.φ)
    end

    @testset "Superbee records ignition times" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        initial_ignited = count(isfinite, grid.t_ignite)
        for _ in 1:10
            advance!(grid, F, 0.5, Superbee())
        end
        @test count(isfinite, grid.t_ignite) > initial_ignited
    end

    @testset "simulate! with solver=Godunov() matches default" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, steps=20, dt=0.5)

        grid2 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, steps=20, dt=0.5, solver=Godunov())

        @test grid1.φ == grid2.φ
    end

    @testset "simulate! with solver=Superbee() runs" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        simulate!(grid, model, steps=20, dt=0.5, solver=Superbee())
        @test burn_area(grid) > 0
    end

    @testset "both solvers converge to similar fire areas" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_god = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_god, 750.0, 750.0, 50.0)
        simulate!(grid_god, model, steps=50, dt=0.5, solver=Godunov())

        grid_sup = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_sup, 750.0, 750.0, 50.0)
        simulate!(grid_sup, model, steps=50, dt=0.5, solver=Superbee())

        area_god = burn_area(grid_god)
        area_sup = burn_area(grid_sup)
        @test area_god > 0
        @test area_sup > 0
        # Both should be in the same order of magnitude
        ratio = area_sup / area_god
        @test 0.3 < ratio < 3.0
    end

    @testset "Superbee with all boundary conditions" begin
        for bc in (ZeroNeumann(), Dirichlet(), Periodic())
            grid = LevelSetGrid(30, 30, dx=30.0, bc=bc)
            ignite!(grid, 450.0, 450.0, 50.0)
            F = fill(10.0, size(grid))
            for _ in 1:5
                advance!(grid, F, 0.5, Superbee())
            end
            @test burn_area(grid) > 0
        end
    end
end

@testset "WENO5 Solver" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "WENO5 runs and produces burned area" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        for _ in 1:10
            advance!(grid, F, 0.5, WENO5())
        end
        @test burn_area(grid) > 0
        @test grid.t ≈ 5.0
    end

    @testset "WENO5 phi clamping" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        solver = WENO5(phi_clamp=100.0)
        for _ in 1:20
            advance!(grid, F, 0.5, solver)
        end
        @test all(x -> -100.0 <= x <= 100.0, grid.φ)
    end

    @testset "WENO5 records ignition times" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        initial_ignited = count(isfinite, grid.t_ignite)
        for _ in 1:10
            advance!(grid, F, 0.5, WENO5())
        end
        @test count(isfinite, grid.t_ignite) > initial_ignited
    end

    @testset "simulate! with solver=WENO5() runs" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        simulate!(grid, model, steps=20, dt=0.5, solver=WENO5())
        @test burn_area(grid) > 0
    end

    @testset "WENO5 with all boundary conditions" begin
        for bc in (ZeroNeumann(), Dirichlet(), Periodic())
            grid = LevelSetGrid(30, 30, dx=30.0, bc=bc)
            ignite!(grid, 450.0, 450.0, 50.0)
            F = fill(10.0, size(grid))
            for _ in 1:5
                advance!(grid, F, 0.5, WENO5())
            end
            @test burn_area(grid) > 0
        end
    end

    @testset "all three solvers converge to similar fire areas" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        areas = Float64[]
        for solver in (Godunov(), Superbee(), WENO5())
            g = LevelSetGrid(50, 50, dx=30.0)
            ignite!(g, 750.0, 750.0, 50.0)
            simulate!(g, model, steps=50, dt=0.5, solver=solver)
            push!(areas, burn_area(g))
        end
        for a in areas
            @test a > 0
            @test 0.3 < a / areas[1] < 3.0
        end
    end
end

@testset "Newton Reinitialization" begin
    @testset "IterativeReinit backward compat" begin
        grid1 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid1))
        for _ in 1:10
            advance!(grid1, F, 0.5)
        end
        reinitialize!(grid1)

        grid2 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        for _ in 1:10
            advance!(grid2, F, 0.5)
        end
        reinitialize!(grid2, IterativeReinit())

        @test grid1.φ == grid2.φ
    end

    @testset "NewtonReinit produces signed distance" begin
        grid = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid, 750.0, 750.0, 200.0)
        F = fill(10.0, size(grid))
        for _ in 1:20
            advance!(grid, F, 0.5)
        end
        reinitialize!(grid, NewtonReinit())

        # Check |∇φ| ≈ 1 at interior points
        φ = grid.φ
        ny, nx = size(φ)
        grad_mags = Float64[]
        for j in 3:nx-2, i in 3:ny-2
            dφdx = (φ[i, j+1] - φ[i, j-1]) / (2 * grid.dx)
            dφdy = (φ[i+1, j] - φ[i-1, j]) / (2 * grid.dy)
            push!(grad_mags, hypot(dφdx, dφdy))
        end
        mean_grad = sum(grad_mags) / length(grad_mags)
        @test 0.5 < mean_grad < 1.5
    end

    @testset "NewtonReinit preserves interface" begin
        grid = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid, 750.0, 750.0, 200.0)
        burned_before = count(<(0), grid.φ)
        reinitialize!(grid, NewtonReinit())
        burned_after = count(<(0), grid.φ)
        @test abs(burned_after - burned_before) <= 5
    end

    @testset "NewtonReinit preserves ignition times" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        F = fill(10.0, size(grid))
        for _ in 1:10
            advance!(grid, F, 0.5)
        end
        t_ig_before = copy(grid.t_ignite)
        reinitialize!(grid, NewtonReinit())
        for j in axes(grid.φ, 2), i in axes(grid.φ, 1)
            if isfinite(t_ig_before[i, j])
                @test grid.t_ignite[i, j] == t_ig_before[i, j]
            end
        end
    end

    @testset "simulate! with reinit=NewtonReinit()" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        simulate!(grid, model, steps=20, dt=0.5, reinit=NewtonReinit())
        @test burn_area(grid) > 0
    end
end

@testset "Curvature Regularization" begin
    M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

    @testset "curvature=0 matches default" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid1, 450.0, 450.0, 50.0)
        simulate!(grid1, model, steps=20, dt=0.5)

        grid2 = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid2, 450.0, 450.0, 50.0)
        simulate!(grid2, model, steps=20, dt=0.5, curvature=0.0)

        @test grid1.φ == grid2.φ
    end

    @testset "curvature > 0 runs and produces burned area" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        simulate!(grid, model, steps=20, dt=0.5, curvature=1.0)
        @test burn_area(grid) > 0
    end

    @testset "curvature produces different fire shape" begin
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid_no = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_no, 750.0, 750.0, 50.0)
        simulate!(grid_no, model, steps=50, dt=0.5, curvature=0.0)

        grid_curv = LevelSetGrid(50, 50, dx=30.0)
        ignite!(grid_curv, 750.0, 750.0, 50.0)
        simulate!(grid_curv, model, steps=50, dt=0.5, curvature=5.0)

        @test burn_area(grid_no) > 0
        @test burn_area(grid_curv) > 0
        @test burn_area(grid_no) != burn_area(grid_curv)
    end

    @testset "cfl_dt with curvature constraint" begin
        grid = LevelSetGrid(100, 100, dx=30.0)
        F = fill(10.0, size(grid))
        dt_no_curv = cfl_dt(grid, F)
        dt_curv = cfl_dt(grid, F; curvature=1.0)
        @test dt_curv <= dt_no_curv
    end
end

@testset "CFL" begin
    @testset "cfl_dt computation" begin
        grid = LevelSetGrid(100, 100, dx=30.0)
        F = fill(10.0, size(grid))
        @test cfl_dt(grid, F) == 0.5 * 30.0 / 10.0  # 1.5
        @test cfl_dt(grid, F; cfl=1.0) == 30.0 / 10.0  # 3.0
    end

    @testset "cfl_dt with zero spread" begin
        grid = LevelSetGrid(10, 10, dx=30.0)
        F = fill(0.0, size(grid))
        @test cfl_dt(grid, F) == Inf
    end

    @testset "simulate! with auto CFL" begin
        grid = LevelSetGrid(100, 100, dx=30.0)
        ignite!(grid, 1500.0, 1500.0, 50.0)
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid, model, steps=20)
        @test burn_area(grid) > 0
        @test grid.t > 0
    end

    @testset "simulate! with explicit dt still works" begin
        grid = LevelSetGrid(30, 30, dx=30.0)
        ignite!(grid, 450.0, 450.0, 50.0)
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
        simulate!(grid, model, steps=10, dt=0.5)
        @test grid.t ≈ 5.0
    end
end
