const ALL_FUELS = [
    SHORT_GRASS, TIMBER_GRASS, TALL_GRASS, CHAPARRAL, BRUSH, DORMANT_BRUSH,
    SOUTHERN_ROUGH, CLOSED_TIMBER_LITTER, HARDWOOD_LITTER, TIMBER_UNDERSTORY,
    LIGHT_SLASH, MEDIUM_SLASH, HEAVY_SLASH,
]

@testset "Rothermel" begin
    @testset "rate_of_spread is positive for all fuel models" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        for fuel in ALL_FUELS
            R = rate_of_spread(fuel; moisture=M, wind=5.0, slope=0.0)
            @test R > 0
        end
    end

    @testset "residence_time is positive for all fuel models" begin
        for fuel in ALL_FUELS
            @test residence_time(fuel) > 0
        end
    end

    @testset "wind increases rate_of_spread" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        R_calm = rate_of_spread(SHORT_GRASS; moisture=M, wind=0.0, slope=0.0)
        R_wind = rate_of_spread(SHORT_GRASS; moisture=M, wind=8.0, slope=0.0)
        @test R_wind > R_calm
    end

    @testset "slope increases rate_of_spread" begin
        M = FuelClasses(d1=0.06, d10=0.07, d100=0.08, herb=0.0, wood=0.0)
        R_flat = rate_of_spread(SHORT_GRASS; moisture=M, wind=0.0, slope=0.0)
        R_slope = rate_of_spread(SHORT_GRASS; moisture=M, wind=0.0, slope=0.5)
        @test R_slope > R_flat
    end
end
