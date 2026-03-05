import Wildfires.CellularAutomata as CA
using Wildfires.CellularAutomata: CAGrid, CellState, UNBURNED, BURNING, BURNED, UNBURNABLE,
    Moore, VonNeumann
using Wildfires.SpreadModels: Trace

@testset "Cellular Automata" begin

    @testset "CAGrid construction" begin
        grid = CAGrid(50, 40, dx=30.0)
        @test size(grid) == (40, 50)
        @test all(==(UNBURNED), grid.state)
        @test all(isinf, grid.t_ignite)
        @test all(isinf, grid.t_arrival)
        @test grid.dx == 30.0
        @test grid.dy == 30.0
        @test grid.t == 0.0
        @test grid.neighborhood isa Moore
    end

    @testset "coordinates" begin
        grid = CAGrid(10, 10, dx=100.0, x0=500.0, y0=200.0)
        xs = CA.xcoords(grid)
        ys = CA.ycoords(grid)
        @test first(xs) ≈ 550.0
        @test last(xs) ≈ 1450.0
        @test first(ys) ≈ 250.0
        @test last(ys) ≈ 1150.0
    end

    @testset "ignite!" begin
        grid = CAGrid(50, 50, dx=30.0)
        CA.ignite!(grid, 750.0, 750.0, 100.0)
        @test any(==(BURNING), grid.state)
        @test count(==(BURNING), grid.state) > 0
        for j in axes(grid.state, 2), i in axes(grid.state, 1)
            if grid.state[i, j] == BURNING
                @test grid.t_ignite[i, j] == 0.0
            end
        end
    end

    @testset "set_unburnable!" begin
        grid = CAGrid(50, 50, dx=30.0)
        CA.set_unburnable!(grid, 750.0, 750.0, 100.0)
        @test any(==(UNBURNABLE), grid.state)
        for j in axes(grid.state, 2), i in axes(grid.state, 1)
            if grid.state[i, j] == UNBURNABLE
                @test isnan(grid.t_ignite[i, j])
            end
        end
    end

    @testset "queries" begin
        grid = CAGrid(20, 20, dx=30.0)
        CA.ignite!(grid, 300.0, 300.0, 50.0)
        @test any(CA.burning(grid))
        @test !any(CA.burned(grid))
        @test all(CA.burnable(grid))
        @test CA.burn_area(grid) > 0
    end

    @testset "VonNeumann neighborhood" begin
        grid = CAGrid(50, 50, dx=30.0, neighborhood=VonNeumann())
        @test grid.neighborhood isa VonNeumann
        CA.ignite!(grid, 750.0, 750.0, 50.0)
        @test any(==(BURNING), grid.state)
    end

    @testset "show" begin
        grid = CAGrid(50, 50, dx=30.0)
        s = sprint(show, MIME"text/plain"(), grid)
        @test contains(s, "CAGrid")
        @test contains(s, "50×50")
    end

    @testset "deterministic simulation" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        # Run simulation twice and compare
        grid1 = CAGrid(50, 50, dx=30.0)
        CA.ignite!(grid1, 750.0, 750.0, 100.0)
        simulate!(grid1, model, steps=50, dt=0.5)

        grid2 = CAGrid(50, 50, dx=30.0)
        CA.ignite!(grid2, 750.0, 750.0, 100.0)
        simulate!(grid2, model, steps=50, dt=0.5)

        @test grid1.state == grid2.state
        @test grid1.t_ignite == grid2.t_ignite
        @test grid1.t == grid2.t
    end

    @testset "isotropic spread (no wind, no slope)" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=0.0), UniformMoisture(M), FlatTerrain())

        grid = CAGrid(80, 80, dx=30.0)
        CA.ignite!(grid, 1200.0, 1200.0, 50.0)
        simulate!(grid, model, steps=100, dt=0.5)

        # Fire should have spread
        @test CA.burn_area(grid) > 50.0^2 * π
        # With no wind, spread should be roughly symmetric
        n_burning_or_burned = count(s -> s == BURNING || s == BURNED, grid.state)
        @test n_burning_or_burned > 0
    end

    @testset "directional spread with wind" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        # Wind from the west (direction=π means FROM west, pushes east = +x direction)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=10.0, direction=π),
            UniformMoisture(M), FlatTerrain(), EllipticalBlending())

        grid = CAGrid(100, 60, dx=30.0)
        cx, cy = 1500.0, 900.0
        CA.ignite!(grid, cx, cy, 50.0)
        simulate!(grid, model, steps=100, dt=0.3)

        # Fire should spread further in +x (downwind) than -x (upwind)
        xs = CA.xcoords(grid)
        ys = CA.ycoords(grid)
        center_row = argmin(abs.(collect(ys) .- cy))

        # Find easternmost and westernmost burned cells in center row
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
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=0.0), UniformMoisture(M), FlatTerrain())

        grid = CAGrid(50, 50, dx=30.0)
        CA.ignite!(grid, 750.0, 750.0, 60.0)
        simulate!(grid, model, steps=200, dt=0.5, residence_time=5.0)

        # With residence_time, initially ignited cells should have transitioned to BURNED
        # (they've been burning for 100 min, well past 5 min)
        @test any(==(BURNED), grid.state)
    end

    @testset "unburnable barrier stops fire" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=5.0, direction=π),
            UniformMoisture(M), FlatTerrain())

        grid = CAGrid(80, 40, dx=30.0)
        CA.ignite!(grid, 600.0, 600.0, 50.0)

        # Create unburnable wall at x=1200 (blocking fire spread to the east)
        xs = CA.xcoords(grid)
        ys = CA.ycoords(grid)
        for j in eachindex(xs), i in eachindex(ys)
            if abs(xs[j] - 1200.0) < 30.0
                grid.state[i, j] = UNBURNABLE
                grid.t_ignite[i, j] = NaN
            end
        end

        simulate!(grid, model, steps=200, dt=0.3)

        # No cells east of the barrier should be burned
        for j in eachindex(xs), i in eachindex(ys)
            if xs[j] > 1230.0
                @test grid.state[i, j] == UNBURNED
            end
        end
    end

    @testset "CFL auto time stepping" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid = CAGrid(40, 40, dx=30.0)
        CA.ignite!(grid, 600.0, 600.0, 60.0)
        # dt=nothing triggers CFL auto stepping
        simulate!(grid, model, steps=50)
        @test grid.t > 0.0
        @test CA.burn_area(grid) > 0
    end

    @testset "Trace" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        grid = CAGrid(40, 40, dx=30.0)
        CA.ignite!(grid, 600.0, 600.0, 60.0)
        trace = Trace(grid, 5)
        simulate!(grid, model, steps=20, dt=0.5, trace=trace)

        # 1 initial + 4 recorded (steps 5, 10, 15, 20)
        @test length(trace.stack) == 5
        @test trace.stack[1][1] == 0.0  # initial time
        @test trace.stack[end][1] == grid.t
        @test trace.stack[1][2] isa Matrix{CellState}
    end

    @testset "burnout scaling reduces spread" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

        # Without burnout scaling
        grid1 = CAGrid(60, 60, dx=30.0)
        CA.ignite!(grid1, 900.0, 900.0, 60.0)
        simulate!(grid1, model, steps=50, dt=0.5)

        # With burnout scaling (should spread less)
        grid2 = CAGrid(60, 60, dx=30.0)
        CA.ignite!(grid2, 900.0, 900.0, 60.0)
        simulate!(grid2, model, steps=50, dt=0.5, burnout=ExponentialBurnout(1.0))

        @test CA.burn_area(grid2) <= CA.burn_area(grid1)
    end

    @testset "Custom cell type" begin
        # Define a minimal custom cell
        struct FuelCell
            state::CellState
            fuel_load::Float64
        end

        CA.cellstate(c::FuelCell) = c.state
        CA.on_ignite(c::FuelCell, t) = FuelCell(BURNING, c.fuel_load)
        CA.on_burnout(c::FuelCell, t) = FuelCell(BURNED, c.fuel_load)
        CA.on_unburnable(c::FuelCell) = FuelCell(UNBURNABLE, 0.0)

        @testset "CAGrid with custom cells" begin
            cells = fill(FuelCell(UNBURNED, 1.0), 50, 50)
            grid = CAGrid(cells, dx=30.0)
            @test size(grid) == (50, 50)
            @test grid.state[1, 1] isa FuelCell
            @test eltype(grid) === FuelCell
        end

        @testset "ignite! with custom cells" begin
            cells = fill(FuelCell(UNBURNED, 1.0), 50, 50)
            grid = CAGrid(cells, dx=30.0)
            CA.ignite!(grid, 750.0, 750.0, 100.0)
            @test any(c -> CA.cellstate(c) == BURNING, grid.state)
            # Fuel load should be preserved
            for c in grid.state
                @test c.fuel_load == 1.0 || c.fuel_load == 0.0
            end
        end

        @testset "queries with custom cells" begin
            cells = fill(FuelCell(UNBURNED, 1.0), 20, 20)
            grid = CAGrid(cells, dx=30.0)
            CA.ignite!(grid, 300.0, 300.0, 50.0)
            @test any(CA.burning(grid))
            @test !any(CA.burned(grid))
            @test all(CA.burnable(grid))
            @test CA.burn_area(grid) > 0
        end

        @testset "simulate! with custom cells and RothermelModel" begin
            M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
            model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0),
                UniformMoisture(M), FlatTerrain())

            cells = fill(FuelCell(UNBURNED, 1.0), 50, 50)
            grid = CAGrid(cells, dx=30.0)
            CA.ignite!(grid, 750.0, 750.0, 100.0)
            simulate!(grid, model, steps=50, dt=0.5)

            @test any(c -> CA.cellstate(c) == BURNING || CA.cellstate(c) == BURNED, grid.state)
            @test grid.t > 0.0
            # Fuel load preserved through simulation
            for c in grid.state
                @test c.fuel_load == 1.0
            end
        end

        @testset "set_unburnable! with custom cells" begin
            cells = fill(FuelCell(UNBURNED, 1.0), 50, 50)
            grid = CAGrid(cells, dx=30.0)
            CA.set_unburnable!(grid, 750.0, 750.0, 100.0)
            @test any(c -> CA.cellstate(c) == UNBURNABLE, grid.state)
        end

        @testset "Trace with custom cells" begin
            M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
            model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0),
                UniformMoisture(M), FlatTerrain())

            cells = fill(FuelCell(UNBURNED, 1.0), 40, 40)
            grid = CAGrid(cells, dx=30.0)
            CA.ignite!(grid, 600.0, 600.0, 60.0)
            trace = Trace(grid, 5)
            simulate!(grid, model, steps=20, dt=0.5, trace=trace)

            @test length(trace.stack) == 5
            @test trace.stack[1][2] isa Matrix{FuelCell}
        end

        @testset "show with custom cells" begin
            cells = fill(FuelCell(UNBURNED, 1.0), 30, 30)
            grid = CAGrid(cells, dx=30.0)
            s = sprint(show, MIME"text/plain"(), grid)
            @test contains(s, "FuelCell")
        end
    end

end
