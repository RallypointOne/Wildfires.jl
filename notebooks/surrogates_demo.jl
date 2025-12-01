### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 8e9ed22e-bd5c-11f0-b5b5-87b42c61bf8c
begin
	using Pkg 
	Pkg.activate(joinpath(@__DIR__, ".."))
	using Wildfires
	using GlobalGrids
	using Extents
	using GeoMakie, GLMakie
	using Tyler
	using TileProviders
	using MapTiles
	using Landfire
	using DataFrames
	using ProgressMeter
	using Scratch
	using Rasters
	using SQLite
	using DBInterface
	using Surrogates
	
	import GeoInterface as GI
	import GeometryOps as GO
	import GeoFormatTypes as GFT
	import GeoJSON

	Rasters.checkmem!(false)

	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 606495d1-4819-4230-9fac-a85299fdc29e
md"""
# Marshall Fire

"""

# ╔═╡ 0b02db56-1be0-44c0-90e7-5ad486bd13f0
marshall = get(Data.MarshallFireFinalPerimeter())

# ╔═╡ 59cbc841-b78c-4c72-8beb-83d54277ca25
ext = Extents.grow(GI.extent(marshall), 0.1)

# ╔═╡ d719a59f-e8f5-4edc-9533-b54215c13e34
md"""
# Raster Mosaic

- Data downloaded from USGS Data Download App
- Need a little bit of processing to:
  - Change projection
  - Remove `missing`
"""

# ╔═╡ 17122be9-a35a-46fc-a0df-175d2d3fec72
files = [
	"USGS_1M_13_x47y442_CO_DRCOG_2020_B20.tif",
	"USGS_1M_13_x47y443_CO_DRCOG_2020_B20.tif",
	"USGS_one_meter_x48y442_CO_SoPlatteRiver_Lot5_2013.tif",
	"USGS_1M_13_x48y443_CO_DRCOG_2020_B20.tif"
]

# ╔═╡ 39914adb-dcdf-43f7-b797-b00878438de1
rasters = map(files) do file 
	r = Raster(joinpath(@__DIR__, "..", "data", file), checkmem=false)
	r = resample(r, crs=GFT.EPSG("EPSG:4326"))
	# r = replace_missing(r, missingval=0.0)
	crop(r, to=Extents.grow(ext, .1))
end

# ╔═╡ f306d000-ab4b-4b3f-8f21-199710cf990c
r = Float32.(crop(mosaic(first, rasters), to=ext))

# ╔═╡ 099bcc3d-16ab-4c62-92ce-d8d7a06d94b4
let 
	f, a, p = plot(r)
	poly!(a, marshall, color=(:red, 0.3))
	f
end

# ╔═╡ 8afc8b96-ce94-4413-b0b7-86f2b13cd818
md"## Subset"

# ╔═╡ 02f3e125-00a9-45c8-aa7a-1869a0d13929
# Subset
r2 = r[1:250, (1:250) .+ 250]

# ╔═╡ 253ad85a-44d4-4daa-983c-adb5ce7b9576
plot(r2)

# ╔═╡ 9e4cf389-e874-433b-ac7b-eeb142dcb777
md"""
# Surrogates.jl
"""

# ╔═╡ db330a0a-74d8-41f1-96ca-fea798543652
ex2 = GI.extent(r2)

# ╔═╡ 67e00548-c3a1-4d9b-8bc2-99d548cfd0d5
lb = [ex2.X[1], ex2.Y[1]]

# ╔═╡ d67bfda2-6ec9-47f7-89b1-3a66c10f1844
ub = [ex2.X[2], ex2.Y[2]]

# ╔═╡ c2c6cbbf-4843-46a3-b994-ff286a9d6ab4
xys = sample(2000, lb, ub, SobolSample())

# ╔═╡ 65554707-7942-4777-b579-b9412dab2994
z = [r2[X(Near(x)), Y(Near(y))] for (x,y) in xys]

# ╔═╡ a4d4a9e4-8f32-4531-ae09-fdaf58f44402
scatter(xys)

# ╔═╡ 6bb3b0c3-3f71-4408-885d-0e98e8c3c4a8
surr2 = AbstractGPSurrogate(xys, z)

# ╔═╡ 11895aad-fba0-4f8b-80f8-06777616550d
z

# ╔═╡ 53665ceb-2627-43c6-a6c4-e4d05e11063f
surr = Kriging(xys, z, lb, ub)
# surr = AbstractGPSurrogate(xys, z)

# ╔═╡ a3aeac39-d092-41fc-b59c-e4cee1bd2a81
let 
	fig = Figure()
	ax = Axis(fig[1,1])
	ax2 = Axis(fig[1,2])
	linkaxes!(ax, ax2)
	
	plot!(ax, r2)
	scatter!(ax2, xys; color = :black)

	xrng = r2.dims[1].val
	yrng = r2.dims[2].val
	Z = [surr([x, y]) for x in xrng, y in yrng]
	heatmap!(ax2, xrng, yrng, Z)
	hidedecorations!(ax)
	hidedecorations!(ax2)

	fig
end

# ╔═╡ 8805f9ce-55c7-4540-ba56-1f35ad4319c8
md"## Data Processing"

# ╔═╡ 37c4bb7f-7069-49be-b47a-e74194203a5d
normalize(x) = 2 .* (x .- minimum(x)) ./ (maximum(x) - minimum(x)) .- 1

# ╔═╡ Cell order:
# ╟─8e9ed22e-bd5c-11f0-b5b5-87b42c61bf8c
# ╟─606495d1-4819-4230-9fac-a85299fdc29e
# ╠═0b02db56-1be0-44c0-90e7-5ad486bd13f0
# ╠═59cbc841-b78c-4c72-8beb-83d54277ca25
# ╟─d719a59f-e8f5-4edc-9533-b54215c13e34
# ╟─17122be9-a35a-46fc-a0df-175d2d3fec72
# ╟─39914adb-dcdf-43f7-b797-b00878438de1
# ╠═f306d000-ab4b-4b3f-8f21-199710cf990c
# ╟─099bcc3d-16ab-4c62-92ce-d8d7a06d94b4
# ╟─8afc8b96-ce94-4413-b0b7-86f2b13cd818
# ╠═02f3e125-00a9-45c8-aa7a-1869a0d13929
# ╠═253ad85a-44d4-4daa-983c-adb5ce7b9576
# ╟─9e4cf389-e874-433b-ac7b-eeb142dcb777
# ╟─db330a0a-74d8-41f1-96ca-fea798543652
# ╠═67e00548-c3a1-4d9b-8bc2-99d548cfd0d5
# ╠═d67bfda2-6ec9-47f7-89b1-3a66c10f1844
# ╠═65554707-7942-4777-b579-b9412dab2994
# ╠═c2c6cbbf-4843-46a3-b994-ff286a9d6ab4
# ╠═a4d4a9e4-8f32-4531-ae09-fdaf58f44402
# ╠═6bb3b0c3-3f71-4408-885d-0e98e8c3c4a8
# ╠═11895aad-fba0-4f8b-80f8-06777616550d
# ╠═53665ceb-2627-43c6-a6c4-e4d05e11063f
# ╠═a3aeac39-d092-41fc-b59c-e4cee1bd2a81
# ╟─8805f9ce-55c7-4540-ba56-1f35ad4319c8
# ╠═37c4bb7f-7069-49be-b47a-e74194203a5d
