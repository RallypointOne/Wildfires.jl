using Test
using Wildfires
using Rasters, ArchGDAL
using Oceananigans
using Oceananigans.OutputReaders: Time
using Oceananigans.Grids: znode, xnodes, ynodes, topology, Center, Face, Flat
using Breeze.TerrainFollowingDiscretization

@testset "Wildfires.jl" begin
    @test Base.get_extension(Wildfires, :WildfiresRastersExt) isa Module

    include("test_georeference.jl")
    include("test_fire_model.jl")
    include("test_rothermel.jl")
    include("test_hamada.jl")
    include("test_polygons.jl")
    include("test_structures.jl")
    include("test_spread.jl")
end
