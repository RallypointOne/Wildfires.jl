using Wildfires
using Wildfires.Rothermel: FuelClasses, Rothermel, rate_of_spread, residence_time,
    SHORT_GRASS, TIMBER_GRASS, TALL_GRASS, CHAPARRAL, BRUSH, DORMANT_BRUSH,
    SOUTHERN_ROUGH, CLOSED_TIMBER_LITTER, HARDWOOD_LITTER, TIMBER_UNDERSTORY,
    LIGHT_SLASH, MEDIUM_SLASH, HEAVY_SLASH,
    NFFL_MODELS, nffl_model
using Wildfires.LevelSet: LevelSetGrid, xcoords, ycoords, ignite!, advance!, reinitialize!, burned, burn_area,
    cfl_dt, AbstractBoundaryCondition, ZeroNeumann, Dirichlet, Periodic, set_unburnable!, burnable,
    AbstractSolver, Godunov, Superbee, WENO5,
    AbstractReinitMethod, IterativeReinit, NewtonReinit
using Wildfires.SpreadModels: RothermelModel, UniformWind, UniformMoisture, FlatTerrain,
    UniformSlope, DynamicMoisture, spread_rate_field!, simulate!, update!,
    CosineBlending, EllipticalBlending, length_to_breadth, fire_eccentricity,
    NoBurnout, ExponentialBurnout, LinearBurnout,
    NoBurnin, ExponentialBurnin, LinearBurnin,
    directional_speed, AbstractFuel
using Adapt
using KernelAbstractions, GPUArraysCore
using Test

@testset "Wildfires.jl" begin
    include("test_rothermel.jl")
    include("test_levelset.jl")
    include("test_components.jl")
    include("test_cellular_automata.jl")

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
            model = RothermelModel(SHORT_GRASS, UniformWind(speed=8.0), UniformMoisture(M), FlatTerrain())
            simulate!(grid, model, steps=20, dt=0.5)
            @test burn_area(grid) > 0
            @test grid.t == 10.0
        end
    end

    include("test_pinn.jl")
end
