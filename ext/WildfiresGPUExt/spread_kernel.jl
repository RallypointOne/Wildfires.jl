#-----------------------------------------------------------------------------# GPU spread_rate_field! — compute on CPU, upload to GPU

# Helper: create a CPU copy of the model's moisture for scalar indexing
_cpu_moisture(m::SpreadModel.AbstractMoisture) = m
function _cpu_moisture(m::SpreadModel.DynamicMoisture)
    SpreadModel.DynamicMoisture(
        Array(m.d1), m.base, m.ambient_d1,
        m.dry_rate, m.recovery_rate, m.min_d1,
        m.dx, m.dy, m.x0, m.y0
    )
end

_cpu_model(model) = model
function _cpu_model(model::SpreadModel.FireSpreadModel)
    SpreadModel.FireSpreadModel(
        model.fuel, model.wind, _cpu_moisture(model.moisture), model.terrain
    )
end

function _gpu_spread_rate_field!(F, model, grid)
    T = eltype(F)
    F_cpu = Matrix{T}(undef, size(F))
    φ_cpu = Array(grid.φ)
    cpu_grid = LevelSetGrid(φ_cpu, Array(grid.t_ignite), grid.dx, grid.dy, grid.x0, grid.y0, grid.t, grid.bc)
    cm = _cpu_model(model)
    SpreadModel.spread_rate_field!(F_cpu, cm, cpu_grid)
    copyto!(F, F_cpu)
    F
end

# Generic model dispatch
function SpreadModel.spread_rate_field!(F::AbstractGPUArray{T, 2}, model, grid::LevelSetGrid) where {T}
    _gpu_spread_rate_field!(F, model, grid)
end

# FireSpreadModel dispatch (resolves ambiguity with base FireSpreadModel method)
function SpreadModel.spread_rate_field!(F::AbstractGPUArray{T, 2}, model::SpreadModel.FireSpreadModel, grid::LevelSetGrid) where {T}
    _gpu_spread_rate_field!(F, model, grid)
end

#-----------------------------------------------------------------------------# GPU simulate!

function SpreadModel.simulate!(grid::LevelSetGrid{T, <:AbstractGPUArray}, model;
                               steps::Int=100, dt=nothing, cfl=0.5, reinit_every::Int=10, burnout=nothing, trace=nothing, progress::Bool=false) where {T}
    F_gpu = similar(grid.φ)

    for step in 1:steps
        SpreadModel.spread_rate_field!(F_gpu, model, grid)
        burnout !== nothing && _gpu_apply_burnout!(F_gpu, grid, burnout)
        step_dt = dt === nothing ? LevelSet.cfl_dt(grid, F_gpu; cfl=cfl) : dt
        SpreadModel.update!(model, grid, step_dt)
        LevelSet.advance!(grid, F_gpu, step_dt)
        step % reinit_every == 0 && LevelSet.reinitialize!(grid)
        trace !== nothing && step % trace.every == 0 && SpreadModel._record!(trace, grid)
        progress && step % max(1, steps ÷ 100) == 0 && SpreadModel._print_progress(step, steps, grid)
    end
    progress && println()
    grid
end

@kernel function _burnout_kernel!(F, @Const(t_ignite), t_now, t_r)
    i, j = @index(Global, NTuple)
    t_ig = t_ignite[i, j]
    if isfinite(t_ig) && t_now - t_ig > t_r
        F[i, j] = zero(eltype(F))
    end
end

function _gpu_apply_burnout!(F, grid, t_r)
    backend = get_backend(F)
    ny, nx = size(F)
    _burnout_kernel!(backend)(F, grid.t_ignite, grid.t, t_r, ndrange=(ny, nx))
    KernelAbstractions.synchronize(backend)
end
