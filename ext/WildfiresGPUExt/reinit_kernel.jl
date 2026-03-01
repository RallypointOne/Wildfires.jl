#-----------------------------------------------------------------------------# GPU reinitialize! kernel

@kernel function _reinit_kernel!(φ, @Const(φ_old), dx, dy, dτ, h, nx, ny, bc)
    i, j = @index(Global, NTuple)

    T = eltype(φ)
    z = zero(T)

    if !LevelSet._skip_update(i, j, ny, nx, bc)
        φ_ij = φ_old[i, j]
        S = LevelSet._smoothed_sign(φ_ij, h)

        dxm = LevelSet._Dxm(φ_old, i, j, dx, bc)
        dxp = LevelSet._Dxp(φ_old, i, j, nx, dx, bc)
        dym = LevelSet._Dym(φ_old, i, j, dy, bc)
        dyp = LevelSet._Dyp(φ_old, i, j, ny, dy, bc)

        if S > z
            a = max(max(dxm, z), -min(dxp, z))
            b = max(max(dym, z), -min(dyp, z))
        else
            a = max(-min(dxm, z), max(dxp, z))
            b = max(-min(dym, z), max(dyp, z))
        end

        grad_mag = hypot(a, b)
        φ[i, j] = φ_ij - dτ * S * (grad_mag - one(T))
    end
end

function LevelSet.reinitialize!(g::LevelSetGrid{T, <:AbstractGPUArray}; iterations::Int=5) where {T}
    φ = g.φ
    ny, nx = size(φ)
    dx, dy = g.dx, g.dy
    h = min(dx, dy)
    dτ = h / 2

    backend = get_backend(φ)
    for _ in 1:iterations
        φ_old = copy(φ)
        _reinit_kernel!(backend)(φ, φ_old, dx, dy, dτ, h, nx, ny, g.bc, ndrange=(ny, nx))
        KernelAbstractions.synchronize(backend)
    end
    g
end
