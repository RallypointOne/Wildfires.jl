"""
Physics-informed wildfire propagation model based on Bottero et al. (2020).

This implementation solves the level-set fire spread equation using finite differences:
    ∂ψ/∂t + S(t,x) * |∇ψ| = 0

where S(t,x) is the fire spread rate depending on wind and slope factors.

Reference: Bottero et al. "Physics-Informed Machine Learning Simulator for Wildfire
Propagation" (2020). Uses a modified Rothermel formula:
    S = R0 * (1 + φW + φS)

where:
  - R0: baseline spread rate without wind or slope [m/s]
  - φW: wind influence factor = C_W * max(0, wind·∇ψ/|∇ψ|)
  - φS: slope influence factor = C_S * max(0, ∇ψ·∇z/(|∇ψ||∇z|)) * |∇z|
"""


using OrdinaryDiffEq
using Printf
using GLMakie
using Colors

"""
    compute_spread_rate(ψx, ψy, x, y, t, wind_model, elev_model, R0, C_W, C_S)

Compute the fire spread rate S(t,x,y) based on the Rothermel formula.

This accounts for:
  1. Baseline spread rate (R0)
  2. Wind influence: φW = C_W * max(0, (wind·∇ψ)/|∇ψ|)
  3. Slope influence: φS = C_S * max(0, (∇ψ·∇z)/(|∇ψ||∇z|)) * |∇z|
"""
function compute_spread_rate(ψx, ψy, x, y, t, wind_model, elev_model, R0, C_W, C_S; eps_psi=1e-6)
    # Gradient magnitude (stabilized)
    grad_psi = sqrt(ψx^2 + ψy^2 + eps_psi^2)

    # Wind factor: φW
    wind = wind_model(t, x, y)
    u_w, v_w = wind[1], wind[2]
    wind_align = (u_w * ψx + v_w * ψy) / grad_psi
    phi_W = C_W * max(0.0, wind_align)

    # Slope factor: φS
    h = 1e-6  # finite difference step
    z_ref = elev_model(t, x, y)
    dz_dx = (elev_model(t, x + h, y) - elev_model(t, x - h, y)) / (2h)
    dz_dy = (elev_model(t, x, y + h) - elev_model(t, x, y - h)) / (2h)

    grad_z = sqrt(dz_dx^2 + dz_dy^2 + eps_psi^2)
    if grad_z > eps_psi
        slope_align = (ψx * dz_dx + ψy * dz_dy) / (grad_psi * grad_z)
    else
        slope_align = 0.0
    end
    phi_S = C_S * max(0.0, slope_align) * grad_z

    # Total spread rate (Rothermel formula)
    S = R0 * (1.0 + phi_W + phi_S)
    return S
end

"""
    bottero_fire_rhs!(dψdt, ψ, p, t)

Right-hand side for the level-set fire spread PDE:
    ∂ψ/∂t + S(t,x) * |∇ψ| = 0

Uses Godunov's upwind scheme for the gradient magnitude.
"""
function bottero_fire_rhs!(dψdt, ψ, p, t)
    wind_model, elev_model, R0, C_W, C_S, xs, ys, dx, dy = p

    nx, ny = size(ψ)

    @inbounds for j in 1:ny
        @inbounds for i in 1:nx
            # Current position
            x = xs[i]
            y = ys[j]

            # Handle boundaries with Neumann (zero-flux)
            ip = min(i + 1, nx)
            im = max(i - 1, 1)
            jp = min(j + 1, ny)
            jm = max(j - 1, 1)

            # Central differences for gradient
            ψx_c = (ψ[ip, j] - ψ[im, j]) / (2dx)
            ψy_c = (ψ[i, jp] - ψ[i, jm]) / (2dy)

            # Godunov upwind scheme for |∇ψ|
            # Forward and backward differences in x
            Dpx = (ψ[ip, j] - ψ[i, j]) / dx
            Dmx = (ψ[i, j] - ψ[im, j]) / dx
            # Forward and backward differences in y
            Dpy = (ψ[i, jp] - ψ[i, j]) / dy
            Dmy = (ψ[i, j] - ψ[i, jm]) / dy

            # Godunov: |∇ψ|² = max(max(D⁻ₓ, 0)², max(-D⁺ₓ, 0)²) + ...
            termx = max(max(Dmx, 0.0)^2, max(-Dpx, 0.0)^2)
            termy = max(max(Dmy, 0.0)^2, max(-Dpy, 0.0)^2)
            eps_psi = 1e-6
            grad_psi = sqrt(termx + termy + eps_psi^2)

            # Compute spread rate
            S = compute_spread_rate(ψx_c, ψy_c, x, y, t, wind_model, elev_model, R0, C_W, C_S)

            # Level-set PDE: ψₜ = -S * |∇ψ|
            dψdt[i, j] = -S * grad_psi
        end
    end

    return nothing
end

"""
    bottero_fire_model(;
        wind_model = (t, x, y) -> (2.0, 1.0),
        elev_model = (t, x, y) -> 0.0,
        extent = ((-1000.0, 1000.0), (-1000.0, 1000.0)),
        grid_resolution = (101, 101),
        ignition_center = (0.0, 0.0),
        ignition_radii = (50.0, 25.0),
        R0 = 0.20,
        C_W = 0.6,
        C_S = 1.0
    ) -> (ψ0, xs, ys, dx, dy, metadata)

Create a level-set fire spread model initialization.

# Arguments
- `wind_model`: Function (t, x, y) -> (u, v) returning wind velocity [m/s]
- `elev_model`: Function (t, x, y) -> z returning elevation [m]
- `extent`: Tuple ((x_min, x_max), (y_min, y_max)) for domain
- `grid_resolution`: Tuple (nx, ny) for discretization
- `ignition_center`: (cx, cy) center of initial fire
- `ignition_radii`: (rx, ry) radii of elliptical ignition
- `R0`: Baseline spread rate [m/s]
- `C_W`: Wind influence coefficient
- `C_S`: Slope influence coefficient

# Returns
- `ψ0`: Initial level set function (nx × ny matrix)
- `xs`, `ys`: Grid coordinates
- `dx`, `dy`: Grid spacing
- `metadata`: Dict with parameters
"""
function bottero_fire_model(;
    wind_model=(t, x, y) -> (2.0, 1.0),
    elev_model=(t, x, y) -> 0.0,
    extent=((-1000.0, 1000.0), (-1000.0, 1000.0)),
    grid_resolution=(101, 101),
    ignition_center=(0.0, 0.0),
    ignition_radii=(50.0, 25.0),
    R0=0.20,
    C_W=0.6,
    C_S=1.0
)
    # Unpack domain
    (xmin, xmax), (ymin, ymax) = extent
    nx, ny = grid_resolution

    # Grid coordinates
    xs = LinRange(xmin, xmax, nx)
    ys = LinRange(ymin, ymax, ny)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # Initial level set: signed distance from ellipse
    # ψ < 0: burnt, ψ = 0: fireline, ψ > 0: unburnt
    cx, cy = ignition_center
    rx, ry = ignition_radii

    ψ0 = zeros(nx, ny)
    @inbounds for j in 1:ny
        @inbounds for i in 1:nx
            x, y = xs[i], ys[j]
            # Ellipse distance formula (approximate)
            ex = (x - cx) / rx
            ey = (y - cy) / ry
            ellipse_dist = sqrt(ex^2 + ey^2)
            # Signed distance
            ψ0[i, j] = (ellipse_dist - 1.0) * max(rx, ry)
        end
    end

    metadata = Dict(
        :extent => extent,
        :grid_resolution => grid_resolution,
        :ignition_center => ignition_center,
        :ignition_radii => ignition_radii,
        :R0 => R0,
        :C_W => C_W,
        :C_S => C_S,
        :xmin => xmin, :xmax => xmax,
        :ymin => ymin, :ymax => ymax
    )

    return ψ0, xs, ys, dx, dy, metadata
end

"""
    plot_fire_evolution(sol, xs, ys, wind_model, meta)

Create an interactive 3D GLMakie visualization of the fire spread evolution.
Shows the level-set function evolution over time with:
  - Fire front (contour at ψ=0) highlighted in red
  - Burned area (ψ<0) in orange
  - Unburnt area (ψ>0) in light colors
"""
function plot_fire_evolution(sol, xs, ys, wind_model, meta)
    println("\nCreating GLMakie visualization...")

    # Select timesteps to visualize
    nsteps = length(sol.t)
    steps_to_plot = range(1, nsteps, length=min(5, nsteps)) |> collect .|> Int

    fig = Figure(size=(1400, 1000))

    # Create subplot for each selected time step
    for (idx, step_idx) in enumerate(steps_to_plot)
        t_current = sol.t[step_idx]
        ψ_current = sol.u[step_idx]

        ax = Axis3(
            fig[ceil(Int, idx/2), mod(idx-1, 2) + 1],
            title="Fire Spread at t=$(round(Int, t_current))s",
            xlabel="x [m]",
            ylabel="y [m]",
            zlabel="ψ [m]",
            elevation = .4pi
        )

        # Convert coordinates to arrays for surface plot
        x_arr = collect(xs)
        y_arr = collect(ys)

        # Plot level set surface with color mapping
        # ψ < -10: fully burned (dark orange)
        # -10 < ψ < 0: burning front (red)
        # 0 < ψ: unburnt (light)
        colors = zeros(RGB, size(ψ_current))
        for i in 1:size(ψ_current, 1)
            for j in 1:size(ψ_current, 2)
                ψ_val = ψ_current[i, j]
                if ψ_val < -10
                    colors[i, j] = RGB(0.8, 0.4, 0.0)  # Dark orange
                elseif ψ_val < 0
                    colors[i, j] = RGB(1.0, 0.3, 0.0)  # Red-orange
                elseif ψ_val < 10
                    colors[i, j] = RGB(1.0, 0.8, 0.2)  # Yellow
                else
                    colors[i, j] = RGB(0.8, 1.0, 0.8)  # Light green
                end
            end
        end

        surface!(ax, x_arr, y_arr, ψ_current, color=ψ_current .< 0, shading=NoShading)

        # Add contour for fireline (ψ = 0)
        contour!(ax, x_arr, y_arr, ψ_current, levels=[0.0], color=:red, linewidth=3)

        # Add wind arrow
        wind = wind_model(t_current, 0.0, 0.0)
        u_w, v_w = wind[1], wind[2]
        wind_mag = sqrt(u_w^2 + v_w^2)
        if wind_mag > 0
            arrow_scale = 200.0
            arrows2d!(ax,
                [meta[:xmin]], [meta[:ymin]], [0.0],
                [u_w * arrow_scale], [v_w * arrow_scale], [0.0],
                color=:blue)
        end
    end

    supertitle = fig[0, :] = Label(fig, "Wildfire Propagation via Level Set Method",
        fontsize=20, font=:bold)

    display(fig)

    return fig
end


# ============================================================================
# Example: Simple idealized fire spread scenario
# ============================================================================


println("Bottero et al. (2020) Fire Spread Model")
println("=" ^ 70)

# Wind and elevation models
wind_model = (t, x, y) -> (2.0, 1.0)  # Constant 2 m/s E, 1 m/s N
elev_model = (t, x, y) -> 0.0         # Flat terrain

# Create fire model
ψ0, xs, ys, dx, dy, meta = bottero_fire_model(
    wind_model=wind_model,
    elev_model=elev_model,
    extent=((-1000.0, 1000.0), (-1000.0, 1000.0)),
    grid_resolution=(101, 101),
    ignition_center=(0.0, 0.0),
    ignition_radii=(50.0, 25.0),
    R0=0.20,
    C_W=0.6,
    C_S=1.0
)

nx, ny = size(ψ0)
println("Grid: $nx × $ny")
println("Domain: [$(meta[:xmin]), $(meta[:xmax])] × [$(meta[:ymin]), $(meta[:ymax])]")
println("Spacing: dx=$dx, dy=$dy")
println("Ignition: center=$(meta[:ignition_center]), radii=$(meta[:ignition_radii])")
println("Rothermel: R0=$(meta[:R0]) m/s, C_W=$(meta[:C_W]), C_S=$(meta[:C_S])")
println()

# Set up ODE problem
p = (wind_model, elev_model, meta[:R0], meta[:C_W], meta[:C_S], xs, ys, dx, dy)
tspan = (0.0, 500.0)
prob = ODEProblem(bottero_fire_rhs!, ψ0, tspan, p)

# Solve
println("Solving PDE with tspan = $tspan...")
sol = solve(prob, SSPRK33(), dt=1.0, adaptive=false, progress=true)

println("Completed $(length(sol.t)) time steps")
println()

# Final statistics
final_ψ = sol.u[end]
burned_cells = count(x -> x ≤ 0, final_ψ)
burned_area = burned_cells * dx * dy
perimeter_cells = count(i -> -1 ≤ final_ψ[i] ≤ 1, eachindex(final_ψ))

println("Final State at t=$(sol.t[end])s:")
println("  Burned cells: $burned_cells / $(nx*ny)")
println("  Burned area: $(burned_area/1e4) hectares")
println("  Perimeter cells: $perimeter_cells")
println("  Level set: min=$(minimum(final_ψ)), max=$(maximum(final_ψ))")
println()

# Show time evolution of burned area
println("Burned area evolution:")
for i in 1:min(10, length(sol.u))
    t = sol.t[i]
    burned = count(x -> x ≤ 0, sol.u[i])
    area = burned * dx * dy
    println("  t=$(@sprintf "%.1f" t)s: $(burned) cells, $(area/1e4) hectares")
end
println()

# Create visualization
fig = plot_fire_evolution(sol, xs, ys, wind_model, meta)
println("Visualization complete! Window should appear...")
