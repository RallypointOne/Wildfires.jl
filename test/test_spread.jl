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

    normal = NormalProjection()
    spread_rate!(speed, model, bed, M; shape = normal)
    @test all(isfinite, interior(speed))
    @test all(interior(speed) .≈ R₀)      # no wind, no slope: isotropic

    spread_rate!(speed, model, bed, M; wind = (U, 0.0), shape = normal)
    @test at(speed, east) ≈ R_wind rtol = 1e-2   # normal ≈ (1, 0.025)
    @test at(speed, west) == R₀                   # upwind: negative component is calm
    @test R₀ < at(speed, north) < R_wind          # crosswind: small component
    @test all(interior(speed) .>= R₀)
    @test all(isfinite, interior(speed))

    spread_rate!(speed, model, bed, M; slope = (S, 0.0), shape = normal)
    @test at(speed, east) ≈ R_slope rtol = 1e-2
    @test at(speed, west) == R₀

    # Fields for wind and slope give the same result as numbers.
    u = Field{Center, Center, Nothing}(grid)
    ∂x_h = Field{Center, Center, Nothing}(grid)
    set!(u, U)
    set!(∂x_h, S)
    spread_rate!(speed, model, bed, M; wind = (U, 0.0), slope = (S, 0.0), shape = normal)
    expected = copy(interior(speed))
    spread_rate!(speed, model, bed, M; wind = (u, 0.0), slope = (∂x_h, 0.0), shape = normal)
    @test interior(speed) == expected

    # The front moves further downwind than upwind.
    for _ in 1:20
        spread_rate!(speed, model, bed, M; wind = (U, 0.0), shape = normal)
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
    spread_rate!(speed32, model32, bed, M; wind = (U, 0.0), slope = (S, 0.0), shape = normal)
    @test eltype(speed32) == Float32
    @test all(isfinite, interior(speed32))
    @test maximum(interior(speed32)) ≈ spread_rate(bed, M, waf * U, S) rtol = 5e-2
end

@testset "HuygensEllipse" begin
    @test length_to_breadth(0.0) == 1
    @test length_to_breadth(2.0) ≈ 2.7 rtol = 0.05
    @test length_to_breadth(50.0) == 8
    @test length_to_breadth(50.0; max_length_to_breadth = 10) == 10
    @test length_to_breadth(1f0) isa Float32

    grid = RectilinearGrid(size = (64, 64), x = (-320.0, 320.0), y = (-320.0, 320.0),
                           topology = (Bounded, Bounded, Flat))
    model = FireModel(grid)
    ignite!(model, 0.0, 0.0; radius = 100.0)
    speed = Field{Center, Center, Nothing}(grid)
    bed = FuelBed(Wildfires.NFFL.SHORT_GRASS)
    M = FuelClasses(0.08, 1.0)
    U = 5.0
    waf = wind_adjustment(bed)
    east = (findfirst(≈(205.0), xnodes(grid, Center())), findfirst(≈(5.0), ynodes(grid, Center())))
    west = (findfirst(≈(-205.0), xnodes(grid, Center())), east[2])
    north = (east[2], east[1])
    at(f, ij) = interior(f)[ij..., 1]

    # Wind only: head, flank, and back follow the Anderson ellipse.
    spread_rate!(speed, model, bed, M; wind = (U, 0.0))          # default shape
    head = spread_rate(bed, M, waf * U, 0.0)
    LB = length_to_breadth(waf * U)
    e = sqrt(1 - 1 / LB^2)
    a = head / (1 + e)
    flank, back = a / LB, a * (1 - e)
    @test at(speed, east) ≈ head rtol = 1e-2
    @test at(speed, west) ≈ back rtol = 2e-2
    # The north cell's normal is (5, 205)/|·|, 1.4° off the flank direction,
    # where the support function already exceeds the flank rate by 5%.
    nx, ny = 5 / hypot(5, 205), 205 / hypot(5, 205)
    @test at(speed, north) ≈ ellipse_speed(nx, ny, 1.0, 0.0, head, flank, back) rtol = 1e-2
    @test at(speed, north) > 2 * spread_rate(bed, M, 0.0, 0.0)   # flanks well above the no-wind rate

    # Calm: a circle at the no-wind rate.
    spread_rate!(speed, model, bed, M)
    @test all(interior(speed) .≈ spread_rate(bed, M, 0.0, 0.0))

    # Slope only: the upslope head equals the Rothermel slope rate, since the
    # equivalent wind reproduces the slope factor exactly.
    S = 0.5
    spread_rate!(speed, model, bed, M; slope = (S, 0.0))
    @test at(speed, east) ≈ spread_rate(bed, M, 0.0, S) rtol = 1e-2
    @test at(speed, west) < at(speed, east)
    @test all(isfinite, interior(speed)) && all(interior(speed) .> 0)

    # Wind and slope add as vectors: opposing them slows the head.
    spread_rate!(speed, model, bed, M; wind = (U, 0.0), slope = (-S, 0.0))
    @test at(speed, east) < head

    # Explicit max ratio narrows the flanks.
    spread_rate!(speed, model, bed, M; wind = (20.0, 0.0), shape = HuygensEllipse(max_length_to_breadth = 2))
    wide = at(speed, north)
    spread_rate!(speed, model, bed, M; wind = (20.0, 0.0), shape = HuygensEllipse(max_length_to_breadth = 8))
    @test at(speed, north) < wide
end

@testset "FuelMap and time series wind" begin
    grid = RectilinearGrid(size = (64, 64), x = (-320.0, 320.0), y = (-320.0, 320.0),
                           topology = (Bounded, Bounded, Flat))
    model = FireModel(grid)
    ignite!(model, 0.0, 0.0; radius = 100.0)
    speed = Field{Center, Center, Nothing}(grid)
    M = FuelClasses(0.08, 1.0)
    U = 5.0

    # Grass west of x = 0, timber litter east, and an unknown code in one strip.
    code(x, y) = x < 0 ? 1 : (x > 250 ? 91 : 8)
    fuel = FuelMap(grid, code, Wildfires.NFFL)
    @test fuel.codes == [1:13; 0]
    @test length(fuel.beds) == 14
    @test Set(interior(fuel.index)) == Set([1.0, 8.0, 14.0])
    spread_rate!(speed, model, fuel, M; wind = (U, 0.0))
    s = interior(speed)[:, :, 1]
    @test all(s[end-3:end, :] .== 0)                       # unknown code: non-burnable
    grass = FuelBed(Wildfires.NFFL.SHORT_GRASS)
    litter = FuelBed(Wildfires.NFFL.CLOSED_TIMBER_LITTER)
    spread_rate!(speed, model, grass, M; wind = (U, 0.0))
    @test s[1:32, :] == interior(speed)[1:32, :, 1]
    spread_rate!(speed, model, litter, M; wind = (U, 0.0))
    @test s[33:56, :] == interior(speed)[33:56, :, 1]

    # A field of codes works too, and SB40 tables.
    codes = Field{Center, Center, Nothing}(grid)
    set!(codes, code)
    @test interior(FuelMap(grid, codes, Wildfires.NFFL).index) == interior(fuel.index)
    @test FuelMap(grid, code, Wildfires.SB40).codes[end] == 0
    @test_throws ArgumentError FuelMap(grid, code, [Wildfires.NFFL.SHORT_GRASS, Wildfires.NFFL.SHORT_GRASS])

    # Remapping the developed class onto short grass makes the east strip burn as grass.
    urban = FuelMap(grid, code, Wildfires.NFFL; remap = (91 => 1,))
    @test Set(interior(urban.index)) == Set([1.0, 8.0])
    spread_rate!(speed, model, urban, M; wind = (U, 0.0))
    remapped = copy(interior(speed))
    spread_rate!(speed, model, grass, M; wind = (U, 0.0))
    @test remapped[end-3:end, :, 1] == interior(speed)[end-3:end, :, 1]
    @test_throws ArgumentError FuelMap(grid, code, Wildfires.NFFL; remap = (91 => 77,))
    @test spread_rate(FuelBed(Wildfires.NONBURNABLE), M, U, 0.5) == 0

    # Wind from a FieldTimeSeries is read at the model's clock time.
    u = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 100.0])
    set!(u[1], U)
    set!(u[2], 3U)
    spread_rate!(speed, model, grass, M; wind = (2U, 0.0))
    expected = copy(interior(speed))
    model.clock.time = 50.0
    spread_rate!(speed, model, grass, M; wind = (u, 0.0))
    @test interior(speed) ≈ expected
    model.clock.time = 500.0                                # clamped to the last snapshot
    spread_rate!(speed, model, grass, M; wind = (3U, 0.0))
    expected = copy(interior(speed))
    spread_rate!(speed, model, grass, M; wind = (u, 0.0))
    @test interior(speed) ≈ expected
end
