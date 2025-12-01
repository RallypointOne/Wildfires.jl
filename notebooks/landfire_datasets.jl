### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b4683bbc-bf11-11f0-a5d2-670bd7d3e7a1
begin
	using Pkg 
	Pkg.activate(joinpath(@__DIR__, ".."))
	
	using Wildfires, GeoMakie, GLMakie, Rasters, ArchGDAL, Scratch, Statistics
	using PlutoUI, Landfire
	PlutoUI.TableOfContents()
end

# ╔═╡ d9858a4f-030e-408b-b98a-a396bd39d5c4
md"# Load Data"

# ╔═╡ f5c76e37-f5c1-4359-8253-2e008b3f86e6
begin
	landfire_data = Scratch.get_scratch!(Wildfires, "landfire_data")
	landfire_file = filter(endswith(".tif"), readdir(landfire_data; join=true))[1]
	landfire = Raster(landfire_file, checkmem = false)
	landfire_bands = [landfire[Band=i].refdims[1][1] for i in 1:size(landfire, 3)]
	landfire_dict = Dict(band => landfire[Band=i] for (i, band) in enumerate(landfire_bands))
end

# ╔═╡ 5beae535-d153-4a02-a8d6-630f950aa6c5
products = Landfire.products()

# ╔═╡ 1a633924-613a-4aca-af05-212df73b3b4a
md"# Data Viewer"

# ╔═╡ fe23073a-8515-4648-8c30-99215d2f5e8c
@bind key Select(sort(collect(keys(landfire_dict))))

# ╔═╡ ca396d33-de2d-4413-9276-0c631e63100f
let 
	i = findfirst(x -> x.layer_name == key, products)
	isnothing(i) ? 
		@info("LANDFIRE product not found") :
		products[i]
end

# ╔═╡ db0993a0-660a-41eb-8691-786fea209e25
r = landfire_dict[key];

# ╔═╡ 48fc4961-adf0-4003-944b-7a710a9cd4e1
let 
	fig = Figure(size=(700,600))
	ax = Axis(fig[1,1], aspect=DataAspect())
	# ax2 = Axis(fig[2, 1], aspect=DataAspect())
	plot!(ax, r)
	# plot!(ax2, rdiff)
	fig
end

# ╔═╡ Cell order:
# ╟─b4683bbc-bf11-11f0-a5d2-670bd7d3e7a1
# ╟─d9858a4f-030e-408b-b98a-a396bd39d5c4
# ╟─f5c76e37-f5c1-4359-8253-2e008b3f86e6
# ╟─5beae535-d153-4a02-a8d6-630f950aa6c5
# ╟─1a633924-613a-4aca-af05-212df73b3b4a
# ╟─fe23073a-8515-4648-8c30-99215d2f5e8c
# ╟─48fc4961-adf0-4003-944b-7a710a9cd4e1
# ╟─ca396d33-de2d-4413-9276-0c631e63100f
# ╟─db0993a0-660a-41eb-8691-786fea209e25
