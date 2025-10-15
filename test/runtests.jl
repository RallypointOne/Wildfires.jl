using Wildfires
using Test

@testset "H3 Hexagons" begin
    for res in 0:15
        o = Cell("Carrboro, NC", res)
        @test is_cell(o)
        @test resolution(o) == res
        @test length(coordinates(o)) == 7
    end
end
