#-----------------------------------------------------------------------------# GPU advance! kernel

@kernel function _advance_kernel!(φ, @Const(φ_old), @Const(F), dx, dy, dt, nx, ny)
    i, j = @index(Global, NTuple)

    T = eltype(φ)
    z = zero(T)

    Fij = F[i, j]
    if Fij > z
        # Upwind finite differences (Godunov)
        Dxm = j > 1  ? (φ_old[i, j] - φ_old[i, j-1]) / dx : z
        Dxp = j < nx ? (φ_old[i, j+1] - φ_old[i, j]) / dx : z
        Dym = i > 1  ? (φ_old[i, j] - φ_old[i-1, j]) / dy : z
        Dyp = i < ny ? (φ_old[i+1, j] - φ_old[i, j]) / dy : z

        Dxm_plus = max(Dxm, z)
        Dxp_minus = min(Dxp, z)
        Dym_plus = max(Dym, z)
        Dyp_minus = min(Dyp, z)

        grad_sq = max(Dxm_plus, -Dxp_minus)^2 + max(Dym_plus, -Dyp_minus)^2
        if grad_sq > z
            grad_mag = sqrt(grad_sq)
            φ[i, j] = φ_old[i, j] - dt * Fij * grad_mag
        end
    end
end

function LevelSet.advance!(g::LevelSetGrid{T, <:AbstractGPUArray}, F::AbstractGPUArray, dt) where {T}
    φ = g.φ
    ny, nx = size(φ)
    φ_old = copy(φ)

    backend = get_backend(φ)
    _advance_kernel!(backend)(φ, φ_old, F, g.dx, g.dy, dt, nx, ny, ndrange=(ny, nx))
    KernelAbstractions.synchronize(backend)

    g.t += dt
    g
end
