### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ fed5a11a-bf2c-11f0-bd94-875b0e3b4dfe
begin
	using Pkg 
	Pkg.activate(joinpath(@__DIR__, ".."))
	
	using Wildfires, GeoMakie, GLMakie, Rasters, ArchGDAL, PlutoUI, Extents, LinearAlgebra
	import GeoInterface as GI
	import GeoFormatTypes as GFT
	import GeometryOps as GO

	Rasters.checkmem!(false)

	PlutoUI.TableOfContents()
end

# ╔═╡ ea3572c3-4a90-448b-b4a7-dfc9c28c978a
md"# Marshall Fire"

# ╔═╡ c97c47d4-0760-4024-b5ce-9a26c735fd3f
marshall = get(Data.MarshallFireFinalPerimeter())

# ╔═╡ b62dca8e-4371-4e17-b1f3-73a1abd041db
ext = Extents.grow(GI.extent(marshall), 0.1)

# ╔═╡ eea20301-09d4-4d56-8db1-0ced7bce2880
md"# Get Raster"

# ╔═╡ 1d9e1d17-5715-4edd-a56c-a1e6012aab20
files = [
	"USGS_1M_13_x47y442_CO_DRCOG_2020_B20.tif",
	"USGS_1M_13_x47y443_CO_DRCOG_2020_B20.tif",
	"USGS_one_meter_x48y442_CO_SoPlatteRiver_Lot5_2013.tif",
	"USGS_1M_13_x48y443_CO_DRCOG_2020_B20.tif"
]

# ╔═╡ 91c6a0b1-2697-4cad-9d93-dfc06ece6e18
rasters = map(files) do file 
	r = Raster(joinpath(@__DIR__, "..", "data", file), checkmem=false)
end

# ╔═╡ ea996293-54d9-44fb-adbf-f773c635f2f3
r = let 
	rm = mosaic(first, rasters)
	r = resample(rm, crs=GFT.EPSG("EPSG:4326"))
	crop(r, to=ext)
end

# ╔═╡ ad197a1d-ae8c-40ce-a931-66190ed0e097
plot(r)

# ╔═╡ 7a300721-5750-4409-bcc0-e8a3faebffff


# ╔═╡ cf383f42-d6e3-408b-a51f-5ecf4785778e
md"# INR: Vector Data"

# ╔═╡ ef86bf0e-f243-4830-bf9d-698dfc474d1d
let 
	shape = GI.Polygon([[(0.0, 0.0), (0.0, 1.0), (1.0, 1.5), (1.0, 0.0), (0.0, 0.0)]])

	elevation = 0.2pi
	
	fig = Figure() 
	ax = Axis3(fig[1,1]; elevation)
	ax2 = Axis3(fig[1,2]; elevation)
	xlims!(ax, (-1.0, 2.0))
	ylims!(ax, (-1.0, 2.5))
	xlims!(ax2, (-1.0, 2.0))
	ylims!(ax2, (-1.0, 2.5))

	x = range(-1, 2, length=200)
	y = range(-1, 2.5, length=200)

	f(x, y) = GO.contains(shape, (x,y))

	approx_step(dist, k) = tanh(k * dist) / 2 + 1/2
	
	hidedecorations!(ax)
	hidedecorations!(ax2)
	hidespines!(ax)
	hidespines!(ax2)


	f2(x, y) = max(1.0 - approx_step(GO.signed_distance((x,y), shape), 15), 0.0)
	
	surface!(ax, x, y, f)
	surface!(ax2, x, y, f2)
	fig
end

# ╔═╡ Cell order:
# ╟─fed5a11a-bf2c-11f0-bd94-875b0e3b4dfe
# ╟─ea3572c3-4a90-448b-b4a7-dfc9c28c978a
# ╟─c97c47d4-0760-4024-b5ce-9a26c735fd3f
# ╟─b62dca8e-4371-4e17-b1f3-73a1abd041db
# ╟─eea20301-09d4-4d56-8db1-0ced7bce2880
# ╟─1d9e1d17-5715-4edd-a56c-a1e6012aab20
# ╟─91c6a0b1-2697-4cad-9d93-dfc06ece6e18
# ╟─ea996293-54d9-44fb-adbf-f773c635f2f3
# ╠═ad197a1d-ae8c-40ce-a931-66190ed0e097
# ╠═7a300721-5750-4409-bcc0-e8a3faebffff
# ╟─cf383f42-d6e3-408b-a51f-5ecf4785778e
# ╠═ef86bf0e-f243-4830-bf9d-698dfc474d1d
