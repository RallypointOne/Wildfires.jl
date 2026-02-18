#-----------------------------------------------------------------------------# GPU reinitialize! kernel

@kernel function _reinit_kernel!(φ, @Const(φ_old), dx, dy, dτ, nx, ny)
    i, j = @index(Global, NTuple)

    T = eltype(φ)
    z = zero(T)

    S = sign(φ_old[i, j])

    Dxm = j > 1  ? (φ_old[i, j] - φ_old[i, j-1]) / dx : z
    Dxp = j < nx ? (φ_old[i, j+1] - φ_old[i, j]) / dx : z
    Dym = i > 1  ? (φ_old[i, j] - φ_old[i-1, j]) / dy : z
    Dyp = i < ny ? (φ_old[i+1, j] - φ_old[i, j]) / dy : z

    if S > z
        a = max(max(Dxm, z), -min(Dxp, z))
        b = max(max(Dym, z), -min(Dyp, z))
    else
        a = max(-min(Dxm, z), max(Dxp, z))
        b = max(-min(Dym, z), max(Dyp, z))
    end

    grad_mag = hypot(a, b)
    φ[i, j] = φ_old[i, j] - dτ * S * (grad_mag - one(T))
end

function LevelSet.reinitialize!(g::LevelSetGrid{T, <:AbstractGPUArray}; iterations::Int=5) where {T}
    φ = g.φ
    ny, nx = size(φ)
    dx, dy = g.dx, g.dy
    dτ = min(dx, dy) * T(0.5)

    backend = get_backend(φ)
    for _ in 1:iterations
        φ_old = copy(φ)
        _reinit_kernel!(backend)(φ, φ_old, dx, dy, dτ, nx, ny, ndrange=(ny, nx))
        KernelAbstractions.synchronize(backend)
    end
    g
end
