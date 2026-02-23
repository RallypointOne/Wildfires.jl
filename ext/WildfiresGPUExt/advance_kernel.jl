#-----------------------------------------------------------------------------# GPU advance! kernel

@kernel function _advance_kernel!(φ, @Const(φ_old), @Const(F), dx, dy, dt, nx, ny, bc)
    i, j = @index(Global, NTuple)

    T = eltype(φ)
    z = zero(T)

    if !LevelSet._skip_update(i, j, ny, nx, bc)
        Fij = F[i, j]
        if Fij > z
            dxm = LevelSet._Dxm(φ_old, i, j, dx, bc)
            dxp = LevelSet._Dxp(φ_old, i, j, nx, dx, bc)
            dym = LevelSet._Dym(φ_old, i, j, dy, bc)
            dyp = LevelSet._Dyp(φ_old, i, j, ny, dy, bc)

            Dxm_plus = max(dxm, z)
            Dxp_minus = min(dxp, z)
            Dym_plus = max(dym, z)
            Dyp_minus = min(dyp, z)

            grad_sq = max(Dxm_plus, -Dxp_minus)^2 + max(Dym_plus, -Dyp_minus)^2
            if grad_sq > z
                grad_mag = sqrt(grad_sq)
                φ[i, j] = φ_old[i, j] - dt * Fij * grad_mag
            end
        end
    end
end

function LevelSet.advance!(g::LevelSetGrid{T, <:AbstractGPUArray}, F::AbstractGPUArray, dt) where {T}
    φ = g.φ
    ny, nx = size(φ)
    φ_old = copy(φ)

    backend = get_backend(φ)
    _advance_kernel!(backend)(φ, φ_old, F, g.dx, g.dy, dt, nx, ny, g.bc, ndrange=(ny, nx))
    KernelAbstractions.synchronize(backend)

    g.t += dt
    g
end
