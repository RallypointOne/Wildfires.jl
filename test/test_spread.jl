@testset "spread_rate!" begin
    grid = RectilinearGrid(size = (64, 64), x = (-320.0, 320.0), y = (-320.0, 320.0),
                           topology = (Bounded, Bounded, Flat))
    model = FireModel(grid)
    ignite!(model, 0.0, 0.0; radius = 100.0)
    speed = Field{Center, Center, Nothing}(grid)

    bed = FuelBed(Wildfires.NFFL.SHORT_GRASS)
    M = FuelClasses(0.08, 1.0)
    U, S = 5.0, 0.5
    waf = wind_adjustment(bed)
    @test waf ≈ 0.36 atol = 0.01
    R₀ = spread_rate(bed, M, 0.0, 0.0)
    R_wind = spread_rate(bed, M, waf * U, 0.0)
    R_slope = spread_rate(bed, M, 0.0, S)
    @test R₀ < R_wind && R₀ < R_slope

    # Cell centres on the ±x and ±y axes, 200 m from the ignition point.
    east = (findfirst(≈(205.0), xnodes(grid, Center())), findfirst(≈(5.0), ynodes(grid, Center())))
    west = (findfirst(≈(-205.0), xnodes(grid, Center())), east[2])
    north = (east[2], east[1])
    at(f, ij) = interior(f)[ij..., 1]

    spread_rate!(speed, model, bed, M)
    @test all(isfinite, interior(speed))
    @test all(interior(speed) .≈ R₀)      # no wind, no slope: isotropic

    spread_rate!(speed, model, bed, M; wind = (U, 0.0))
    @test at(speed, east) ≈ R_wind rtol = 1e-2   # normal ≈ (1, 0.025)
    @test at(speed, west) == R₀                   # upwind: negative component is calm
    @test R₀ < at(speed, north) < R_wind          # crosswind: small component
    @test all(interior(speed) .>= R₀)
    @test all(isfinite, interior(speed))

    spread_rate!(speed, model, bed, M; slope = (S, 0.0))
    @test at(speed, east) ≈ R_slope rtol = 1e-2
    @test at(speed, west) == R₀

    # Fields for wind and slope give the same result as numbers.
    u = Field{Center, Center, Nothing}(grid)
    ∂x_h = Field{Center, Center, Nothing}(grid)
    set!(u, U)
    set!(∂x_h, S)
    spread_rate!(speed, model, bed, M; wind = (U, 0.0), slope = (S, 0.0))
    expected = copy(interior(speed))
    spread_rate!(speed, model, bed, M; wind = (u, 0.0), slope = (∂x_h, 0.0))
    @test interior(speed) == expected

    # The front moves further downwind than upwind.
    for _ in 1:20
        spread_rate!(speed, model, bed, M; wind = (U, 0.0))
        advance!(model, 1.0; speed)
    end
    φ = interior(model.φ)
    @test φ[east[1] + 8, east[2], 1] < φ[west[1] - 8, west[2], 1]
    @test all(isfinite, φ)

    # Float32 grid with Float64 bed and moisture.
    grid32 = RectilinearGrid(Float32; size = (32, 32), x = (-160, 160), y = (-160, 160),
                             topology = (Bounded, Bounded, Flat))
    model32 = FireModel(grid32)
    ignite!(model32, 0.0, 0.0; radius = 50.0)
    speed32 = Field{Center, Center, Nothing}(grid32)
    spread_rate!(speed32, model32, bed, M; wind = (U, 0.0), slope = (S, 0.0))
    @test eltype(speed32) == Float32
    @test all(isfinite, interior(speed32))
    @test maximum(interior(speed32)) ≈ spread_rate(bed, M, waf * U, S) rtol = 5e-2
end
