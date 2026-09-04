@testset "FuelClasses" begin
    c = FuelClasses(0.06, 0.07, 0.08, 0.6, 0.9)
    @test FuelClasses(0.08, 1.0) == FuelClasses(0.08, 0.08, 0.08, 1.0, 1.0)
    @test FuelClasses(1, 2.0, 3, 4, 5) isa FuelClasses{Float64}
    @test FuelClasses{Float32}(c) isa FuelClasses{Float32}
    @test map(x -> 2x, c) == FuelClasses(0.12, 0.14, 0.16, 1.2, 1.8)
    @test map(+, c, c) == map(x -> 2x, c)
    @test Wildfires.sum_dead(c) ≈ 0.21
    @test Wildfires.sum_live(c) ≈ 1.5
end

@testset "FuelModel and FuelBed" begin
    @test length(Wildfires.NFFL) == 13
    @test length(Wildfires.SB40) == 40
    @test [m.code for m in Wildfires.NFFL] == 1:13
    @test allunique(m.code for m in Wildfires.SB40)

    fm32 = FuelModel{Float32}(Wildfires.NFFL.SHORT_GRASS)
    @test fm32 isa FuelModel{Float32}
    @test fm32.code == 1
    @test FuelBed{Float32}(Wildfires.NFFL.SHORT_GRASS) isa FuelBed{Float32}
    @test isbitstype(FuelBed{Float32})

    # Albini net-load weighting: GS4's live classes share a size bin, so the
    # live net load is their sum; NFFL 10's dead classes are in separate bins,
    # so the dead net load is the surface-area-weighted mean.
    lb = 2000 / 43560
    gs4 = FuelBed(Wildfires.SB40.GS4)
    @test gs4.wn_live ≈ (3.40 + 7.10) * lb * (1 - 0.0555)
    n10 = FuelBed(Wildfires.NFFL.TIMBER_UNDERSTORY)
    w = map(x -> x * lb, Wildfires.NFFL.TIMBER_UNDERSTORY.w)
    @test n10.wn_dead ≈ (n10.f.d1 * w.d1 + n10.f.d10 * w.d10 + n10.f.d100 * w.d100) * (1 - 0.0555)
end

@testset "spread_rate" begin
    chain_per_hour = 66 * 0.3048 / 3600     # → m/s
    mph = 1609.344 / 3600                   # → m/s
    all_models = merge(Wildfires.NFFL, Wildfires.SB40)

    @testset "against BehavePlus (behave library via pyrothermel 0.1.4)" begin
        # Scenario A: 8% dead, 100% live, 5 mi/h midflame, flat (Anderson 1982 conditions).
        # Scenario B: 6/7/8% dead, 60/90% live, 10 mi/h midflame, 30° slope.
        # Static loads, wind limit inactive for these models. behave's NFFL loads
        # round-trip through kg/m² (0.74 → 0.74052), hence the looser NFFL tolerance.
        MA, MB = FuelClasses(0.08, 1.0), FuelClasses(0.06, 0.07, 0.08, 0.6, 0.9)
        ref = (TALL_GRASS = (102.112, 338.9719), CHAPARRAL = (71.4051, 252.2738),
               SOUTHERN_ROUGH = (26.1219, 87.9901), HEAVY_SLASH = (14.1504, 42.6633),
               GR9 = (5.9424, 28.203), GS4 = (3.8361, 14.8233), SH5 = (50.644, 152.2282),
               SH9 = (36.6721, 129.0574), TU3 = (16.8207, 64.3221), TL9 = (7.8507, 27.7587),
               SB4 = (46.3765, 171.6199))
        for (name, (A, B)) in pairs(ref)
            bed = FuelBed(all_models[name])
            rtol = all_models[name].code < 100 ? 5e-3 : 1e-3
            @test spread_rate(bed, MA, 5mph, 0.0) / chain_per_hour ≈ A rtol = rtol
            @test spread_rate(bed, MB, 10mph, tand(30)) / chain_per_hour ≈ B rtol = rtol
        end
    end

    @testset "monotone in wind, slope, moisture" begin
        M = FuelClasses(0.08, 1.0)
        for fm in all_models
            bed = FuelBed(fm)
            R0 = spread_rate(bed, M, 0.0, 0.0)
            @test R0 > 0
            @test spread_rate(bed, M, 2.0, 0.0) > R0
            @test spread_rate(bed, M, 0.0, 0.3) > R0
            @test spread_rate(bed, FuelClasses(0.12, 1.0), 0.0, 0.0) < R0
        end
    end

    @testset "extinction and empty fuel" begin
        bed = FuelBed(Wildfires.NFFL.SHORT_GRASS)
        Mx = Wildfires.NFFL.SHORT_GRASS.Mx
        @test spread_rate(bed, FuelClasses(Mx, 1.0), 2.0, 0.0) == 0
        @test spread_rate(bed, FuelClasses(Mx - 0.01, 1.0), 2.0, 0.0) > 0

        zero5 = FuelClasses(0.0, 0.0, 0.0, 0.0, 0.0)
        empty = FuelBed(FuelModel(99, zero5, zero5, zero5, 0.0, 0.0))
        @test spread_rate(empty, FuelClasses(0.08, 1.0), 5.0, 0.5) == 0
        @test all(isfinite, (empty.Γ, empty.ξ, empty.C_w, empty.C_s, empty.W))
    end

    @testset "negative wind and slope" begin
        bed = FuelBed(Wildfires.NFFL.TALL_GRASS)
        M = FuelClasses(0.08, 1.0)
        @test spread_rate(bed, M, -3.0, 0.0) ≈ spread_rate(bed, M, 0.0, 0.0)
        @test spread_rate(bed, M, 0.0, -0.5) == spread_rate(bed, M, 0.0, 0.0)
    end

    @testset "Float32" begin
        bed = FuelBed{Float32}(Wildfires.SB40.SH5)
        M = FuelClasses(0.06, 0.07, 0.08, 0.6, 0.9)
        R = @inferred spread_rate(bed, M, 4.0, 0.2)
        @test R isa Float32
        @test R ≈ spread_rate(FuelBed(Wildfires.SB40.SH5), M, 4.0, 0.2) rtol = 1e-5
        @test @inferred(spread_rate(bed, FuelClasses{Float32}(M), 4f0, 0.2f0)) === R
    end
end
