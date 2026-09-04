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
