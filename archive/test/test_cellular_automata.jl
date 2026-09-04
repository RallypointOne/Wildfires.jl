@testset "Cellular Automata" begin

    @testset "ca_grid construction" begin
        grid = ca_grid(50, 40; dx=30.0)
        @test size(grid) == (40, 50)
        @test all(==(UNBURNED), grid.state)
        @test all(isinf, grid.layers.t_ignite)
        @test all(isinf, grid.layers.t_arrival)
        @test grid.dx == 30.0
        @test grid.t == 0.0
    end

    @testset "coordinates" begin
        grid = ca_grid(10, 10; dx=100.0, x0=500.0, y0=200.0)
        xs = xcoords(grid)
        ys = ycoords(grid)
        @test first(xs) ≈ 550.0
        @test last(xs) ≈ 1450.0
        @test first(ys) ≈ 250.0
        @test last(ys) ≈ 1150.0
    end

    @testset "ignite!" begin
        grid = ca_grid(50, 50; dx=30.0)
        ignite!(grid, 750.0, 750.0, 100.0)
        @test any(==(BURNING), grid.state)
        @test count(==(BURNING), grid.state) > 0
        for j in axes(grid.state, 2), i in axes(grid.state, 1)
            if grid.state[i, j] == BURNING
                @test grid.layers.t_ignite[i, j] == 0.0
            end
        end
    end

    @testset "set_unburnable!" begin
        grid = ca_grid(50, 50; dx=30.0)
        set_unburnable!(grid, 750.0, 750.0, 100.0)
        @test any(==(UNBURNABLE), grid.state)
        for j in axes(grid.state, 2), i in axes(grid.state, 1)
            if grid.state[i, j] == UNBURNABLE
                @test isnan(grid.layers.t_ignite[i, j])
            end
        end
    end

    @testset "queries" begin
        grid = ca_grid(20, 20; dx=30.0)
        ignite!(grid, 300.0, 300.0, 50.0)
        @test any(burning(grid))
        @test !any(burned(grid))
        @test all(burnable(grid))
        @test burn_area(grid) > 0
    end

    @testset "VonNeumann neighborhood" begin
        prop = CAPropagation(neighborhood=VonNeumann())
        @test prop.neighborhood isa VonNeumann
    end

    @testset "deterministic simulation" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = ca_grid(50, 50; dx=30.0)
        ignite!(grid1, 750.0, 750.0, 100.0)
        simulate!(grid1, model, CAPropagation(); steps=50, dt=0.5)

        grid2 = ca_grid(50, 50; dx=30.0)
        ignite!(grid2, 750.0, 750.0, 100.0)
        simulate!(grid2, model, CAPropagation(); steps=50, dt=0.5)

        @test grid1.state == grid2.state
        @test grid1.layers.t_ignite == grid2.layers.t_ignite
        @test grid1.t == grid2.t
    end

    @testset "isotropic spread (no wind, no slope)" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=0.0), UniformMoisture(M), FlatTerrain())

        grid = ca_grid(80, 80; dx=30.0)
        ignite!(grid, 1200.0, 1200.0, 50.0)
        simulate!(grid, model, CAPropagation(); steps=100, dt=0.5)

        @test burn_area(grid) > 50.0^2 * π
        n_burning_or_burned = count(s -> s == BURNING || s == BURNED, grid.state)
        @test n_burning_or_burned > 0
    end

    @testset "directional spread with wind" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=10.0, direction=π),
            UniformMoisture(M), FlatTerrain(), EllipticalShape())

        grid = ca_grid(100, 60; dx=30.0)
        cx, cy = 1500.0, 900.0
        ignite!(grid, cx, cy, 50.0)
        simulate!(grid, model, CAPropagation(); steps=100, dt=0.3)

        xs = xcoords(grid)
        ys = ycoords(grid)
        center_row = argmin(abs.(collect(ys) .- cy))

        east_max = 0.0
        west_min = Inf
        for j in eachindex(xs)
            if grid.state[center_row, j] == BURNING || grid.state[center_row, j] == BURNED
                east_max = max(east_max, xs[j])
                west_min = min(west_min, xs[j])
            end
        end
        downwind_spread = east_max - cx
        upwind_spread = cx - west_min
        @test downwind_spread > upwind_spread
    end

    @testset "residence_time burnout" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=0.0), UniformMoisture(M), FlatTerrain())

        grid = ca_grid(50, 50; dx=30.0)
        ignite!(grid, 750.0, 750.0, 60.0)
        simulate!(grid, model, CAPropagation(); steps=200, dt=0.5, residence_time=5.0)

        @test any(==(BURNED), grid.state)
    end

    @testset "unburnable barrier stops fire" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=5.0, direction=π),
            UniformMoisture(M), FlatTerrain())

        grid = ca_grid(80, 40; dx=30.0)
        ignite!(grid, 600.0, 600.0, 50.0)

        xs = xcoords(grid)
        ys = ycoords(grid)
        for j in eachindex(xs), i in eachindex(ys)
            if abs(xs[j] - 1200.0) < 30.0
                grid.state[i, j] = UNBURNABLE
                grid.layers.t_ignite[i, j] = NaN
            end
        end

        simulate!(grid, model, CAPropagation(); steps=200, dt=0.3)

        for j in eachindex(xs), i in eachindex(ys)
            if xs[j] > 1230.0
                @test grid.state[i, j] == UNBURNED
            end
        end
    end

    @testset "CFL auto time stepping" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid = ca_grid(40, 40; dx=30.0)
        ignite!(grid, 600.0, 600.0, 60.0)
        simulate!(grid, model, CAPropagation(); steps=50)
        @test grid.t > 0.0
        @test burn_area(grid) > 0
    end

    @testset "Trace" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid = ca_grid(40, 40; dx=30.0)
        ignite!(grid, 600.0, 600.0, 60.0)
        trace = Trace(grid, 5)
        simulate!(grid, model, CAPropagation(); steps=20, dt=0.5, trace=trace)

        # 1 initial + 4 recorded (steps 5, 10, 15, 20)
        @test length(trace.stack) == 5
        @test trace.stack[1][1] == 0.0
        @test trace.stack[end][1] == grid.t
        @test trace.stack[1][2] isa Matrix{CellState}
    end

    @testset "burnout scaling reduces spread" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = FireModel(RothermelROS(SHORT_GRASS), UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid1 = ca_grid(60, 60; dx=30.0)
        ignite!(grid1, 900.0, 900.0, 60.0)
        simulate!(grid1, model, CAPropagation(); steps=50, dt=0.5)

        grid2 = ca_grid(60, 60; dx=30.0)
        ignite!(grid2, 900.0, 900.0, 60.0)
        simulate!(grid2, model, CAPropagation(); steps=50, dt=0.5, burnout=ExponentialBurnout(1.0))

        @test burn_area(grid2) <= burn_area(grid1)
    end

end
