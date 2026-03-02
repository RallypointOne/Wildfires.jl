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
                               steps::Int=100, dt=nothing, cfl=0.5, reinit_every::Int=10, burnout=nothing, burnin=nothing, trace=nothing, progress::Bool=false) where {T}
    bo = SpreadModel._coerce_burnout(burnout)
    bi = SpreadModel._coerce_burnin(burnin)
    F_gpu = similar(grid.φ)

    for step in 1:steps
        SpreadModel.spread_rate_field!(F_gpu, model, grid)
        _gpu_scale_burnout!(F_gpu, grid, bo)
        _gpu_scale_burnin!(F_gpu, grid, bi)
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

_gpu_scale_burnout!(F, grid, ::SpreadModel.NoBurnout) = nothing

@kernel function _exp_burnout_kernel!(F, @Const(t_ignite), t_now, inv_τ)
    i, j = @index(Global, NTuple)
    t_ig = t_ignite[i, j]
    if isfinite(t_ig)
        F[i, j] *= exp(-(t_now - t_ig) * inv_τ)
    end
end

function _gpu_scale_burnout!(F, grid, bo::SpreadModel.ExponentialBurnout)
    backend = get_backend(F)
    ny, nx = size(F)
    _exp_burnout_kernel!(backend)(F, grid.t_ignite, grid.t, inv(bo.τ), ndrange=(ny, nx))
    KernelAbstractions.synchronize(backend)
end

@kernel function _linear_burnout_kernel!(F, @Const(t_ignite), t_now, inv_τ)
    i, j = @index(Global, NTuple)
    t_ig = t_ignite[i, j]
    if isfinite(t_ig)
        t_burning = t_now - t_ig
        F[i, j] *= max(zero(t_burning), one(t_burning) - t_burning * inv_τ)
    end
end

function _gpu_scale_burnout!(F, grid, bo::SpreadModel.LinearBurnout)
    backend = get_backend(F)
    ny, nx = size(F)
    _linear_burnout_kernel!(backend)(F, grid.t_ignite, grid.t, inv(bo.τ), ndrange=(ny, nx))
    KernelAbstractions.synchronize(backend)
end

#-----------------------------------------------------------------------------# GPU burn-in kernels

_gpu_scale_burnin!(F, grid, ::SpreadModel.NoBurnin) = nothing

@kernel function _exp_burnin_kernel!(F, @Const(t_ignite), t_now, inv_τ)
    i, j = @index(Global, NTuple)
    t_ig = t_ignite[i, j]
    if isfinite(t_ig)
        F[i, j] *= one(t_ig) - exp(-(t_now - t_ig) * inv_τ)
    end
end

function _gpu_scale_burnin!(F, grid, bi::SpreadModel.ExponentialBurnin)
    backend = get_backend(F)
    ny, nx = size(F)
    _exp_burnin_kernel!(backend)(F, grid.t_ignite, grid.t, inv(bi.τ), ndrange=(ny, nx))
    KernelAbstractions.synchronize(backend)
end

@kernel function _linear_burnin_kernel!(F, @Const(t_ignite), t_now, inv_τ)
    i, j = @index(Global, NTuple)
    t_ig = t_ignite[i, j]
    if isfinite(t_ig)
        t_burning = t_now - t_ig
        F[i, j] *= min(one(t_burning), t_burning * inv_τ)
    end
end

function _gpu_scale_burnin!(F, grid, bi::SpreadModel.LinearBurnin)
    backend = get_backend(F)
    ny, nx = size(F)
    _linear_burnin_kernel!(backend)(F, grid.t_ignite, grid.t, inv(bi.τ), ndrange=(ny, nx))
    KernelAbstractions.synchronize(backend)
end
