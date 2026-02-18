#-----------------------------------------------------------------------------# GPU DynamicMoisture update! kernel

@kernel function _moisture_update_kernel!(d1, @Const(φ), dry_rate, recovery_rate, ambient_d1, min_d1, dt)
    i, j = @index(Global, NTuple)

    if φ[i, j] > 0  # unburned
        fire_flux = dry_rate / (φ[i, j]^2 + one(eltype(d1)))
        recovery = recovery_rate * (ambient_d1 - d1[i, j])
        d1[i, j] = clamp(d1[i, j] + (-fire_flux + recovery) * dt, min_d1, ambient_d1)
    end
end

function SpreadModel.update!(m::SpreadModel.DynamicMoisture{T, <:AbstractGPUArray}, grid::LevelSetGrid, dt) where {T}
    ny, nx = size(m.d1)
    backend = get_backend(m.d1)
    _moisture_update_kernel!(backend)(
        m.d1, grid.φ, m.dry_rate, m.recovery_rate, m.ambient_d1, m.min_d1, dt,
        ndrange=(ny, nx)
    )
    KernelAbstractions.synchronize(backend)
end
