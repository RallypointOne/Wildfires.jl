"""
Builds a level-set fire spread RHS for ODEProblem.

Returns:
- f!    :: Function  — in-place RHS: f!(du,u,p,t)
- u0    :: Vector    — initial ψ field flattened (ny*nx)
- tspan :: Tuple     — default (0.0, 1.0) (change as needed)
- meta  :: NamedTuple — grid & helpers (x, y, nx, ny, dx, dy, idx, ij)

Usage:
    f!, u0, tspan, meta = level_set_fire_model(...)
    prob = ODEProblem(f!, u0, (0.0, 1800.0))
    sol  = solve(prob, SSPRK33())
"""
function level_set_fire_model(;
        wind_model = (t, x, y) -> (2.0, 1.0),                 # (u,v) m/s
        elev_model = (t, x, y) -> 0.0,                        # z meters
        extent = Extents.Extent(X=(0.0, 2000.0), Y=(0.0, 2000.0)),    # xmin,xmax,ymin,ymax
        ignition_geom = Ellipse((1000.0, 1000.0), (5.0, 2.0)),# (cx,cy),(rx,ry) meters
        nx = 201,
        ny = 201,
        R0 = 0.20,   # baseline rate of spread [m/s]
        CW = 0.6,    # wind influence coefficient
        CS = 1.0,    # slope influence coefficient
        eps_psi = 1e-6,  # small stabilizer for |∇ψ|
        eps_z   = 1e-12  # small stabilizer for |∇z|
    )

    # --- grid ---------------------------------------------------------------
    xmin, xmax = extent.X
    ymin, ymax = extent.Y
    dx = (xmax - xmin) / (nx - 1)
    dy = -(ymax - ymin) / (ny - 1)
    xs = range(xmin, xmax; length = nx)
    ys = range(ymin, ymax; length = ny)

    # linear index helpers (column-major: j is x/column, i is y/row)
    idx(i, j) = i + (j-1)*ny
    function ij(k)
        j = Int(fld(k-1, ny)) + 1
        i = k - (j-1)*ny
        return i, j
    end

    # clamp neighbor indices to implement Neumann (zero-flux) boundaries
    clamp_i(i) = ifelse(i < 1, 1, ifelse(i > ny, ny, i))
    clamp_j(j) = ifelse(j < 1, 1, ifelse(j > nx, nx, j))

    # convenience accessors for wind components
    wind_u(t, x, y) = wind_model(t, x, y)[1]
    wind_v(t, x, y) = wind_model(t, x, y)[2]

    # central-diff elevation slopes at (x,y,t)
    dZdx(t, x, y) = (elev_model(t, x + 0.5dx, y) - elev_model(t, x - 0.5dx, y)) / dx
    dZdy(t, x, y) = (elev_model(t, x, y + 0.5dy) - elev_model(t, x, y - 0.5dy)) / dy

    # --- initial condition ψ0: signed distance of ellipse (<0 inside) -------
    (cx, cy) = ignition_geom.center
    (rx, ry) = ignition_geom.radii
    ψ0(x, y) = sqrt(((x - cx)/rx)^2 + ((y - cy)/ry)^2) - 1.0

    u0 = Vector{Float64}(undef, nx*ny)
    @inbounds for j in 1:nx, i in 1:ny
        u0[idx(i,j)] = ψ0(xs[j], ys[i])
    end

    # --- RHS function: Godunov upwind for |∇ψ|; centered for S's alignment ---
    function f!(du, u, p, t)
        @inbounds for j in 1:nx, i in 1:ny
            # coordinates
            x = xs[j]; y = ys[i]

            # neighbors (clamped at boundaries = Neumann)
            ip = clamp_i(i+1); im = clamp_i(i-1)
            jp = clamp_j(j+1); jm = clamp_j(j-1)

            ψc  = u[idx(i ,j )]
            ψip = u[idx(ip,j )]; ψim = u[idx(im,j )]
            ψjp = u[idx(i ,jp)]; ψjm = u[idx(i ,jm)]

            # one-sided differences
            Dpx = (ψip - ψc )/dx   # forward x
            Dmx = (ψc  - ψim)/dx   # backward x
            Dpy = (ψjp - ψc )/dy   # forward y
            Dmy = (ψc  - ψjm)/dy   # backward y

            # Godunov |∇ψ| for ψ_t + S*|∇ψ| = 0, S >= 0
            termx = max(max(Dmx, 0.0)^2, max(-Dpx, 0.0)^2)
            termy = max(max(Dmy, 0.0)^2, max(-Dpy, 0.0)^2)
            gradG = sqrt(termx + termy + eps_psi^2)

            # centered gradient for alignment computations (smoother)
            ψx_c = (ψjp - ψjm)/(2dy)   # careful: j is x, i is y ⇒ swap? No: ψ(x_j,y_i)
            ψy_c = (ψip - ψim)/(2dx)
            # Wait: index mapping is u[i + (j-1)ny], so j is x, i is y.
            # ψx = dψ/dx uses neighbors in j-direction; ψy uses i-direction:
            ψx_c = (u[idx(i, jp)] - u[idx(i, jm)])/(2dx)
            ψy_c = (u[idx(ip, j)] - u[idx(im, j)])/(2dy)

            gψ_c = sqrt(ψx_c^2 + ψy_c^2 + eps_psi^2)

            # wind alignment factor φ_W
            u_w = wind_u(t, x, y)
            v_w = wind_v(t, x, y)
            wind_align = (u_w*ψx_c + v_w*ψy_c) / gψ_c
            ϕW = CW * max(0.0, wind_align)

            # slope factor φ_S
            Zx = dZdx(t, x, y)
            Zy = dZdy(t, x, y)
            gZ = sqrt(Zx^2 + Zy^2 + eps_z^2)
            align_slope = (ψx_c*Zx + ψy_c*Zy) / (gψ_c * gZ)
            ϕS = CS * max(0.0, align_slope) * gZ

            # total spread rate
            S = R0 * (1 + ϕW + ϕS)

            # PDE: ψ_t = - S * |∇ψ|
            du[idx(i,j)] = - S * gradG
        end
        nothing
    end

    tspan = (0.0, 1.0)  # set your real end time when making the ODEProblem

    meta = (x = xs, y = ys, nx = nx, ny = ny, dx = dx, dy = dy, idx = idx, ij = ij,
            extent = extent, ignition_geom = ignition_geom)

    return f!, u0, tspan, meta
end
