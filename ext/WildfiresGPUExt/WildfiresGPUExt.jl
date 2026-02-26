module WildfiresGPUExt

using Wildfires
using Wildfires.LevelSet: LevelSetGrid, LevelSet, xcoords, ycoords,
    AbstractBoundaryCondition, ZeroNeumann, Dirichlet, Periodic
using Wildfires.SpreadModel: SpreadModel, DynamicMoisture

using Adapt: Adapt
using KernelAbstractions
using GPUArraysCore: AbstractGPUArray

#-----------------------------------------------------------------------------# Adapt.jl integration
Adapt.adapt_structure(to, g::LevelSetGrid) =
    LevelSetGrid(Adapt.adapt(to, g.φ), g.dx, g.dy, g.x0, g.y0, g.t, g.bc)

Adapt.adapt_structure(to, m::DynamicMoisture) =
    DynamicMoisture(Adapt.adapt(to, m.d1), m.base, m.ambient_d1,
        m.dry_rate, m.recovery_rate, m.min_d1, m.dx, m.dy, m.x0, m.y0)

include("advance_kernel.jl")
include("reinit_kernel.jl")
include("moisture_kernel.jl")
include("spread_kernel.jl")

end # module
