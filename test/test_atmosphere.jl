using Breeze: AtmosphereModel, AnelasticDynamics, ReferenceState
using Oceananigans.TimeSteppers: time_step!

@testset "raster_column_series" begin
    Lookups = Rasters.DimensionalData.Lookups
    Δ = 1000.0
    xs = 500_000.0 .+ (0:5) .* Δ
    ys = 4.4e6 .+ (0:5) .* Δ
    dims3 = (X(xs; sampling = Lookups.Intervals(Lookups.Start())),
             Y(ys; sampling = Lookups.Intervals(Lookups.Start())), Band(1:3))
    # Level l holds value l at height 500 l, except one column of heights is
    # given in descending order to check the sort.
    values = Raster([Float64(l) for _ in 1:6, _ in 1:6, l in 1:3], dims3; crs = Rasters.EPSG(32613))
    heights = Raster([500.0 * l for _ in 1:6, _ in 1:6, l in 1:3], dims3; crs = Rasters.EPSG(32613))
    crs = ProjectedCRS(32613, (first(xs) + Δ / 2, first(ys) + Δ / 2))
    grid = RectilinearGrid(size = (4, 4, 8), x = (0.0, 4000.0), y = (0.0, 4000.0), z = (0.0, 2000.0),
                           topology = (Bounded, Bounded, Bounded))
    series = raster_column_series([values, values], [heights, heights], [0.0, 3600.0], grid, crs)
    column = interior(series[1])[2, 2, :]
    z = znodes(grid, Center())                    # 125, 375, ..., 1875
    @test column[1] == 1                          # below the lowest level: held
    @test column[2] == 1                          # 375 m, still below the lowest level
    @test column[3] ≈ 1 + (625 - 500) / 500       # between levels 1 and 2
    @test column[4] ≈ 1 + (875 - 500) / 500
    @test column[end] == 3                        # above the highest level: held
    @test all(interior(series[2]) .== interior(series[1]))
    @test series.times == [0.0, 3600.0]
    @test_throws ArgumentError raster_column_series([values], [heights, heights], [0.0, 1.0], grid, crs)

    # Descending level order gives the same answer.
    flipped = Raster(reverse(parent(values); dims = 3), dims3; crs = Rasters.EPSG(32613))
    flipped_h = Raster(reverse(parent(heights); dims = 3), dims3; crs = Rasters.EPSG(32613))
    @test interior(raster_column_series([flipped], [flipped_h], [0.0], grid, crs)[1]) == interior(series[1])
end

@testset "lateral_sponge" begin
    grid = RectilinearGrid(size = (20, 12, 4), x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Bounded, Bounded, Bounded))
    mask = interior(lateral_sponge(grid; width = 4))[:, :, 1]
    @test all(0 .<= mask .<= 1)
    @test mask[10, 6] == 0 && mask[10, 7] == 0            # interior clear
    @test mask[1, 6] > mask[2, 6] > mask[3, 6] > mask[4, 6] > 0 == mask[5, 6]
    @test mask[1, 6] ≈ mask[20, 6] ≈ mask[10, 1] ≈ mask[10, 12]
    @test mask[1, 1] == mask[1, 6]                        # corners take the maximum
    linear = interior(lateral_sponge(grid; width = 4, ramp = :linear))[:, :, 1]
    @test linear[1, 6] == 1 - 0.5 / 4 && linear[4, 6] == 1 - 3.5 / 4
    @test_throws ArgumentError lateral_sponge(grid; width = 0)
    @test_throws ArgumentError lateral_sponge(grid; ramp = :quadratic)
end

@testset "relaxation_forcing" begin
    # Periodic in the horizontal, so a uniform acceleration is not projected
    # out by the anelastic pressure solve as it would be between walls.
    grid = RectilinearGrid(size = (8, 8, 8), x = (0, 800), y = (0, 800), z = (0, 800),
                           topology = (Periodic, Periodic, Bounded))
    reference = ReferenceState(grid; potential_temperature = 300)

    # A uniform target: u relaxes toward it everywhere at the given rate.
    forcing = (; u = relaxation_forcing(:u, 10.0, 1 / 100))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference), forcing)
    set!(model; θ = 300)
    for _ in 1:10
        time_step!(model, 1.0)
    end
    u = interior(model.velocities.u)
    @test all(u .> 0)
    @test maximum(u) ≈ 10 * (1 - exp(-10 / 100)) rtol = 0.1
    @test maximum(u) ≈ minimum(u)

    # A scalar with a time-series target and the lateral sponge: only the
    # sponge cells move, toward the target interpolated at the clock time.
    target = FieldTimeSeries{Center, Center, Center}(grid, [0.0, 20.0])
    set!(target[1], 300.0)
    set!(target[2], 320.0)
    mask = lateral_sponge(grid; width = 2)
    forcing = (; θ = relaxation_forcing(:θ, target, 1 / 10; mask))
    model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference), forcing)
    set!(model; θ = 300)
    time_step!(model, 1.0)
    θ = interior(model.formulation.potential_temperature)
    early = θ[1, 4, 4] - 300                              # target ≈ 300 at the stage times near t = 0
    @test 0 <= early < 0.1
    @test maximum(abs.(θ[4:5, 4:5, :] .- 300)) < 1e-3      # interior: mask 0
    model.clock.time = 10.0
    time_step!(model, 1.0)
    θ = interior(model.formulation.potential_temperature)
    @test maximum(abs.(θ[4:5, 4:5, :] .- 300)) < 1e-3      # interior: mask 0
    @test θ[1, 4, 4] - 300 > 5 * early                    # sponge cell, target 310 at t = 10
    @test θ[1, 4, 4] ≈ θ[8, 4, 4] ≈ θ[4, 1, 4]            # symmetric sponge
end
