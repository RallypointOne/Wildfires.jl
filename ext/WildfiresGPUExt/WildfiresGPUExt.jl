module WildfiresGPUExt

using Wildfires
using Wildfires.LevelSet: LevelSetGrid, LevelSet, xcoords, ycoords
using Wildfires.SpreadModel: SpreadModel

using KernelAbstractions
using GPUArraysCore: AbstractGPUArray

include("advance_kernel.jl")
include("reinit_kernel.jl")
include("moisture_kernel.jl")
include("spread_kernel.jl")

end # module
