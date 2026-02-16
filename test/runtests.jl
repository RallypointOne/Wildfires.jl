using Wildfires
using Wildfires.Rothermel: FuelClasses, Rothermel, rate_of_spread,
    SHORT_GRASS, TIMBER_GRASS, TALL_GRASS, CHAPARRAL, BRUSH, DORMANT_BRUSH,
    SOUTHERN_ROUGH, CLOSED_TIMBER_LITTER, HARDWOOD_LITTER, TIMBER_UNDERSTORY,
    LIGHT_SLASH, MEDIUM_SLASH, HEAVY_SLASH
using Wildfires.LevelSet: LevelSetGrid, xcoords, ycoords, ignite!, advance!, burned, burn_area
using Wildfires.SpreadModel: FireSpreadModel, UniformWind, UniformMoisture, FlatTerrain, simulate!
using Test

const ALL_FUELS = [
    SHORT_GRASS, TIMBER_GRASS, TALL_GRASS, CHAPARRAL, BRUSH, DORMANT_BRUSH,
    SOUTHERN_ROUGH, CLOSED_TIMBER_LITTER, HARDWOOD_LITTER, TIMBER_UNDERSTORY,
    LIGHT_SLASH, MEDIUM_SLASH, HEAVY_SLASH,
]

@testset "Wildfires.jl" begin

    @testset "FuelClasses" begin
        @testset "keyword constructor" begin
            fc = FuelClasses(d1=1.0, d10=2.0, d100=3.0, herb=4.0, wood=5.0)
            @test fc.d1 == 1.0
            @test fc.d10 == 2.0
            @test fc.d100 == 3.0
            @test fc.herb == 4.0
            @test fc.wood == 5.0
            @test fc isa FuelClasses{Float64}
        end

        @testset "positional constructor with promotion" begin
            fc = FuelClasses(1, 2.0, 3, 4, 5)
            @test fc isa FuelClasses{Float64}
            @test fc.d1 === 1.0
        end

        @testset "map" begin
            fc = FuelClasses(1.0, 2.0, 3.0, 4.0, 5.0)
            fc2 = map(x -> x * 2, fc)
            @test fc2 == FuelClasses(2.0, 4.0, 6.0, 8.0, 10.0)

            fc3 = map(+, fc, fc2)
            @test fc3 == FuelClasses(3.0, 6.0, 9.0, 12.0, 15.0)
        end

        @testset "sum" begin
            fc = FuelClasses(1.0, 2.0, 3.0, 4.0, 5.0)
            @test sum(fc) == 15.0
            @test sum(x -> x^2, fc) == 1.0 + 4.0 + 9.0 + 16.0 + 25.0
        end

        @testset "show" begin
            fc = FuelClasses(1.0, 2.0, 3.0, 4.0, 5.0)
            s = sprint(show, fc)
            @test contains(s, "FuelClasses{Float64}")
            @test contains(s, "d1=1.0")
        end
    end

    @testset "Rothermel struct" begin
        @testset "kwdef constructor" begin
            r = Rothermel(
                name = "Test",
                w  = FuelClasses(1.0, 2.0, 3.0, 4.0, 5.0),
                σ  = FuelClasses(100.0, 100.0, 100.0, 100.0, 100.0),
                h  = FuelClasses(8000.0, 8000.0, 8000.0, 8000.0, 8000.0),
                δ  = 1.0,
                Mx = 0.12,
            )
            @test r isa Rothermel{Float64}
            @test r.name == "Test"
            @test r.w isa FuelClasses{Float64}
            @test r.δ == 1.0
            @test r.Mx == 0.12
        end

        @testset "show" begin
            s = sprint(show, SHORT_GRASS)
            @test contains(s, "Rothermel{")
            @test contains(s, "Short grass")
        end
    end

    @testset "NFFL fuel model constants" begin
        @test length(ALL_FUELS) == 13
        @test all(f -> f isa Rothermel{Float64}, ALL_FUELS)
        @test all(f -> f.δ > 0, ALL_FUELS)
        @test all(f -> 0 < f.Mx ≤ 1, ALL_FUELS)

        @testset "known fuel bed depths" begin
            @test SHORT_GRASS.δ == 1.0
            @test TALL_GRASS.δ == 2.5
            @test CHAPARRAL.δ == 6.0
            @test CLOSED_TIMBER_LITTER.δ == 0.2
            @test HEAVY_SLASH.δ == 3.0
        end
    end

    @testset "rate_of_spread" begin
        M_dead = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        M_live = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.60, wood=0.90)

        @testset "reference values" begin
            @test rate_of_spread(SHORT_GRASS, moisture=M_dead, wind=8.0, slope=0.0) ≈ 31.119 atol=0.01
            @test rate_of_spread(SHORT_GRASS, moisture=M_dead, wind=0.0, slope=0.0) ≈ 1.4035 atol=0.001
            @test rate_of_spread(TIMBER_GRASS, moisture=M_live, wind=8.0, slope=0.0) ≈ 13.513 atol=0.01
            @test rate_of_spread(CHAPARRAL, moisture=M_live, wind=8.0, slope=0.0) ≈ 27.133 atol=0.01
            @test rate_of_spread(CLOSED_TIMBER_LITTER, moisture=M_dead, wind=8.0, slope=0.0) ≈ 0.6755 atol=0.001
            @test rate_of_spread(HEAVY_SLASH, moisture=M_dead, wind=8.0, slope=0.0) ≈ 5.352 atol=0.01
        end

        @testset "all models produce positive ROS" begin
            for fuel in ALL_FUELS
                M = sum(fuel.w) > 0 ? M_live : M_dead
                R = rate_of_spread(fuel, moisture=M, wind=8.0, slope=0.0)
                @test R > 0
            end
        end

        @testset "wind increases spread" begin
            R0 = rate_of_spread(SHORT_GRASS, moisture=M_dead, wind=0.0, slope=0.0)
            R8 = rate_of_spread(SHORT_GRASS, moisture=M_dead, wind=8.0, slope=0.0)
            @test R8 > R0
        end

        @testset "slope increases spread" begin
            R_flat  = rate_of_spread(SHORT_GRASS, moisture=M_dead, wind=8.0, slope=0.0)
            R_slope = rate_of_spread(SHORT_GRASS, moisture=M_dead, wind=8.0, slope=0.4)
            @test R_slope > R_flat
            @test R_slope ≈ 40.361 atol=0.01
        end

        @testset "moisture at/above extinction → zero spread" begin
            M_sat = FuelClasses(d1=0.50, d10=0.50, d100=0.50, herb=0.0, wood=0.0)
            R = rate_of_spread(SHORT_GRASS, moisture=M_sat, wind=8.0, slope=0.0)
            @test R == 0.0
        end

        @testset "higher moisture → lower spread" begin
            M_dry = FuelClasses(d1=0.03, d10=0.04, d100=0.05, herb=0.0, wood=0.0)
            M_wet = FuelClasses(d1=0.10, d10=0.10, d100=0.10, herb=0.0, wood=0.0)
            R_dry = rate_of_spread(SHORT_GRASS, moisture=M_dry, wind=8.0, slope=0.0)
            R_wet = rate_of_spread(SHORT_GRASS, moisture=M_wet, wind=8.0, slope=0.0)
            @test R_dry > R_wet
        end

        @testset "zero fuel depth → zero spread" begin
            zero_fuel = Rothermel(
                name = "zero",
                w  = FuelClasses(0.0, 0.0, 0.0, 0.0, 0.0),
                σ  = FuelClasses(3500.0, 109.0, 30.0, 1500.0, 1500.0),
                h  = FuelClasses(8000.0, 8000.0, 8000.0, 8000.0, 8000.0),
                δ  = 0.0,
                Mx = 0.12,
            )
            R = rate_of_spread(zero_fuel, moisture=M_dead, wind=8.0, slope=0.0)
            @test R == 0.0
        end
    end

    @testset "PINN" begin
        using Lux, ComponentArrays, ForwardDiff, Zygote, Optimization, OptimizationOptimisers
        using Random: MersenneTwister

        @testset "PINNConfig defaults" begin
            config = PINNConfig()
            @test config.hidden_dims == [64, 64, 64]
            @test config.activation == :tanh
            @test config.n_interior == 5000
            @test config.max_epochs == 5000
            @test config.learning_rate == 1e-3
        end

        @testset "PINNSolution show" begin
            sol = PINNSolution(nothing, nothing, nothing, PINNConfig(), Float64[], (tspan=(0,1), xspan=(0,1), yspan=(0,1), phi_scale=1.0), nothing)
            s = sprint(show, sol)
            @test contains(s, "PINNSolution")
        end

        @testset "train_pinn with constant F" begin
            rng = MersenneTwister(42)

            # Small grid, constant spread rate
            grid = LevelSetGrid(20, 20, dx=50.0)
            ignite!(grid, 500.0, 500.0, 100.0)

            # Constant spread model: F = 5.0 m/min everywhere
            const_model = (t, x, y) -> 5.0

            config = PINNConfig(
                hidden_dims = [32, 32],
                n_interior = 100,
                n_boundary = 40,
                max_epochs = 200,
                resample_every = 0,
                learning_rate = 1e-3,
            )

            sol = train_pinn(grid, const_model, (0.0, 10.0);
                             config=config, rng=rng, verbose=false)

            @test sol isa PINNSolution
            @test length(sol.loss_history) >= 200
            @test sol.loss_history[end] < sol.loss_history[1]

            # Hard IC constraint: exact fit at t=0
            grid_ic = LevelSetGrid(20, 20, dx=50.0)
            ignite!(grid_ic, 500.0, 500.0, 100.0)
            predict_on_grid!(grid_ic, sol, 0.0)
            @test grid_ic.φ ≈ grid.φ atol=1e-10

            # Evaluate single point
            φ_val = sol(0.0, 500.0, 500.0)
            @test φ_val isa Float64

            # predict_on_grid returns correct shape
            φ_mat = predict_on_grid(sol, grid, 5.0)
            @test size(φ_mat) == size(grid)

            # predict_on_grid! updates grid
            grid_copy = LevelSetGrid(20, 20, dx=50.0)
            predict_on_grid!(grid_copy, sol, 5.0)
            @test grid_copy.t == 5.0
            @test grid_copy.φ == φ_mat
        end
    end

end
