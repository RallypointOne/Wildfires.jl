using Wildfires
using Wildfires.Rothermel: FuelClasses, Rothermel, rate_of_spread, residence_time,
    SHORT_GRASS, TIMBER_GRASS, TALL_GRASS, CHAPARRAL, BRUSH, DORMANT_BRUSH,
    SOUTHERN_ROUGH, CLOSED_TIMBER_LITTER, HARDWOOD_LITTER, TIMBER_UNDERSTORY,
    LIGHT_SLASH, MEDIUM_SLASH, HEAVY_SLASH
using Wildfires.LevelSet: LevelSetGrid, xcoords, ycoords, ignite!, advance!, reinitialize!, burned, burn_area,
    cfl_dt, AbstractBoundaryCondition, ZeroNeumann, Dirichlet, Periodic, set_unburnable!, burnable
using Wildfires.SpreadModel: FireSpreadModel, UniformWind, UniformMoisture, FlatTerrain,
    UniformSlope, DynamicMoisture, spread_rate_field!, simulate!, update!,
    CosineBlending, EllipticalBlending, length_to_breadth, fire_eccentricity,
    NoBurnout, ExponentialBurnout, LinearBurnout,
    NoBurnin, ExponentialBurnin, LinearBurnin
using Adapt
using KernelAbstractions, GPUArraysCore
using Test

const ALL_FUELS = [
    SHORT_GRASS, TIMBER_GRASS, TALL_GRASS, CHAPARRAL, BRUSH, DORMANT_BRUSH,
    SOUTHERN_ROUGH, CLOSED_TIMBER_LITTER, HARDWOOD_LITTER, TIMBER_UNDERSTORY,
    LIGHT_SLASH, MEDIUM_SLASH, HEAVY_SLASH,
]

@testset "Wildfires.jl" begin

    @testset "GPU Extension" begin
        @testset "extension loaded" begin
            ext = Base.get_extension(Wildfires, :WildfiresGPUExt)
            @test ext !== nothing
        end

        @testset "type parameterization" begin
            grid = LevelSetGrid(20, 20, dx=30.0)
            @test grid isa LevelSetGrid{Float64, Matrix{Float64}}
            @test grid.φ isa Matrix{Float64}
            @test grid.t_ignite isa Matrix{Float64}
        end

        @testset "Adapt roundtrip" begin
            grid = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid, 450.0, 450.0, 50.0)
            set_unburnable!(grid, 300.0, 300.0, 50.0)
            grid2 = Adapt.adapt(Array, grid)
            @test grid2 isa LevelSetGrid{Float64, Matrix{Float64}}
            @test grid2.φ == grid.φ
            @test all(isnan.(grid2.t_ignite) .== isnan.(grid.t_ignite))
            @test grid2.dx == grid.dx
            @test grid2.t == grid.t
        end

        @testset "Adapt DynamicMoisture roundtrip" begin
            grid = LevelSetGrid(20, 20, dx=30.0)
            M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
            dm = DynamicMoisture(grid, M)
            dm2 = Adapt.adapt(Array, dm)
            @test dm2.d1 == dm.d1
            @test dm2.base == dm.base
            @test dm2.ambient_d1 == dm.ambient_d1
        end

        @testset "broadcast ignite!" begin
            grid = LevelSetGrid(50, 50, dx=30.0)
            ignite!(grid, 750.0, 750.0, 100.0)
            @test count(<(0), grid.φ) > 0
            # Check signed distance property: center should be most negative
            cx_idx = round(Int, (750.0 - grid.x0) / grid.dx + 0.5)
            cy_idx = round(Int, (750.0 - grid.y0) / grid.dy + 0.5)
            @test grid.φ[cy_idx, cx_idx] < -50.0
        end

        @testset "simulate! uses similar (not Matrix)" begin
            grid = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid, 450.0, 450.0, 50.0)
            M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
            simulate!(grid, model, steps=20, dt=0.5)
            @test burn_area(grid) > 0
            @test grid.t == 10.0
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
            @test config.max_epochs == 10000
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

    @testset "NeuralPDE PINN" begin
        using NeuralPDE, ModelingToolkit, OptimizationOptimJL

        @testset "NeuralPDEConfig defaults" begin
            config = NeuralPDEConfig()
            @test config.hidden_dims == [16, 16]
            @test config.activation == :σ
            @test config.strategy == :grid
            @test config.grid_step == 0.1
            @test config.max_epochs == 1000
            @test config.optimizer == :lbfgs
            @test config.learning_rate == 1e-2
        end

        @testset "NeuralPDEConfig PINNSolution show" begin
            sol = PINNSolution(nothing, nothing, nothing, NeuralPDEConfig(), Float64[], (tspan=(0,1), xspan=(0,1), yspan=(0,1), phi_scale=1.0), nothing)
            s = sprint(show, sol)
            @test contains(s, "PINNSolution{NeuralPDE}")
        end

        @testset "extension loaded" begin
            ext = Base.get_extension(Wildfires, :WildfiresNeuralPDEExt)
            @test ext !== nothing
        end

        @testset "train_pinn dispatch" begin
            # Verify dispatch routes to NeuralPDE extension (not Lux)
            grid = LevelSetGrid(5, 5, dx=200.0)
            ignite!(grid, 500.0, 500.0, 100.0)
            config = NeuralPDEConfig(hidden_dims=[4], max_epochs=2, grid_step=1.0)
            m = methods(train_pinn, (typeof(grid), Function, Tuple{Float64,Float64}, typeof(config)))
            @test length(m) == 1
            @test occursin("NeuralPDE", string(first(m)))
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
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
            @test model.directional isa CosineBlending
        end

        @testset "EllipticalBlending construction" begin
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain(), EllipticalBlending())
            @test model.directional isa EllipticalBlending
            @test model.directional.formula == :anderson
        end

        @testset "elliptical simulate! runs" begin
            grid = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid, 450.0, 450.0, 50.0)
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain(), EllipticalBlending())
            simulate!(grid, model, steps=20, dt=0.5)
            @test burn_area(grid) > 0
        end

        @testset "slope-only produces elongated fire with elliptical" begin
            grid = LevelSetGrid(80, 80, dx=20.0)
            ignite!(grid, 800.0, 800.0, 50.0)
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=0.0), UniformMoisture(M),
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
            model_cos = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

            grid_ell = LevelSetGrid(50, 50, dx=30.0)
            ignite!(grid_ell, 750.0, 750.0, 50.0)
            model_ell = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain(), EllipticalBlending())

            simulate!(grid_cos, model_cos, steps=30, dt=0.5)
            simulate!(grid_ell, model_ell, steps=30, dt=0.5)

            # Both burn, but produce different perimeters
            @test burn_area(grid_cos) > 0
            @test burn_area(grid_ell) > 0
            @test burn_area(grid_ell) != burn_area(grid_cos)
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
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
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
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
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
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
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

    @testset "Burnout" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)

        @testset "burnout=nothing preserves current behavior" begin
            grid1 = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid1, 450.0, 450.0, 50.0)
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
            simulate!(grid1, model, steps=20, dt=0.5)

            grid2 = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid2, 450.0, 450.0, 50.0)
            simulate!(grid2, model, steps=20, dt=0.5, burnout=nothing)

            @test grid1.φ == grid2.φ
        end

        @testset "NoBurnout() matches burnout=nothing" begin
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

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
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

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
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

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
            model = FireSpreadModel(CHAPARRAL, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

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
            model = FireSpreadModel(CHAPARRAL, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

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
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

            grid1 = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid1, 450.0, 450.0, 50.0)
            simulate!(grid1, model, steps=20, dt=0.5)

            grid2 = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid2, 450.0, 450.0, 50.0)
            simulate!(grid2, model, steps=20, dt=0.5, burnin=nothing)

            @test grid1.φ == grid2.φ
        end

        @testset "NoBurnin() matches burnin=nothing" begin
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

            grid1 = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid1, 450.0, 450.0, 50.0)
            simulate!(grid1, model, steps=20, dt=0.5, burnin=nothing)

            grid2 = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid2, 450.0, 450.0, 50.0)
            simulate!(grid2, model, steps=20, dt=0.5, burnin=NoBurnin())

            @test grid1.φ == grid2.φ
        end

        @testset "ExponentialBurnin limits fire spread" begin
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

            grid_no = LevelSetGrid(50, 50, dx=30.0)
            ignite!(grid_no, 750.0, 750.0, 50.0)
            simulate!(grid_no, model, steps=200, dt=0.5)

            grid_bi = LevelSetGrid(50, 50, dx=30.0)
            ignite!(grid_bi, 750.0, 750.0, 50.0)
            simulate!(grid_bi, model, steps=200, dt=0.5, burnin=ExponentialBurnin(0.5))

            @test burn_area(grid_bi) <= burn_area(grid_no)
        end

        @testset "LinearBurnin limits fire spread" begin
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

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
            model = FireSpreadModel(CHAPARRAL, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())

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
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
            simulate!(grid, model, steps=20)
            @test burn_area(grid) > 0
            @test grid.t > 0
        end

        @testset "simulate! with explicit dt still works" begin
            grid = LevelSetGrid(30, 30, dx=30.0)
            ignite!(grid, 450.0, 450.0, 50.0)
            M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
            model = FireSpreadModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
            simulate!(grid, model, steps=10, dt=0.5)
            @test grid.t ≈ 5.0
        end
    end

end
