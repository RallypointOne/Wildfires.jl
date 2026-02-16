#-----------------------------------------------------------------------------# Ellipse
struct Ellipse
    center::Tuple{Float64, Float64}
    radii::Tuple{Float64, Float64}
end

function level_set_fire_model(;
        wind_model = (t, x, y) -> (2.0, 1.0),
        elev_model = (t, x, y) -> 0.0,
        extent = Extents.Extent(X=(0.0, 2000.0), Y=(0.0, 2000.0)),
        ignition_geom = Ellipse((1000.0, 1000.0), (5.0, 2.0)),
        nx = 201,
        ny = 201,
        R0 = 0.20, # Baseline rate-of-spread (without wind/slope) [meters/second]
        CW = 0.6,  # Wind influence coefficient (dimensionless)
        CS = 1.0,  # Slope influence coefficient (dimensionless)
        eps_psi = 1e-6,
        eps_z = 1e-12,
    )
    # Grid:
    xmin, xmax = extent.X
    ymin, ymax = extent.Y
    dx = (xmax - xmin) / (nx - 1)
    dy = -(ymax - ymin) / (ny - 1)
    xs = range(xmin, xmax; length = nx)
    ys = range(ymin, ymax; length = ny)

    # wind_u(t, x, y) = wind_model(t, x, y)[1]
    # wind_v(t, x, y) = wind_model(t, x, y)[2]

end
