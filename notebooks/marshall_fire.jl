### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ 11225dea-88f0-11f0-2be4-8733351417a6
begin 
	using Revise
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	using Wildfires, GLMakie, GeoMakie, PlutoUI, DataFrames
	
	import GeoInterface as GI 
	import GeoFormatTypes as GFT 
	import GeometryOps as GO
	
	PlutoUI.TableOfContents()
end

# ╔═╡ 0728afb9-8c0b-4ad1-8e7c-ab672d751d65
md"""
# Load Final Perimeter GeoJSON
"""

# ╔═╡ 473b4ad2-5719-40df-a325-13bcf08061f0
marshall = Wildfires.Data.MarshallFireFinalPerimeter().data

# ╔═╡ dbfd7610-620b-41ce-afa2-28b4bd359706
linestring1 = GI.LineString(marshall[1][1])

# ╔═╡ 60373761-5342-4f89-9507-44d09cc92ffd
md"""
# Original Multipolygon
"""

# ╔═╡ 20b9c27e-3896-468b-9206-176da753b4cd
poly(marshall)

# ╔═╡ 6435f263-f671-4029-aa64-880ad596d453
md"# H3 Representation at different resolutions"

# ╔═╡ e88be00f-56ad-4a07-bb85-f2feb2bddb70
let 
	fig = Figure()
	ax = Axis(fig[1,1], title="Marshall Fire Perimeter in H3")

	function f!(i; kw...)
		c = Wildfires.cells(linestring1, i)
		poly!(ax, c; kw...)
		lines!(ax, c; alpha=.3, linewidth=1, color=:black)
	end
	# f!(8, color=:yellow)
	# f!(9, color=:orange)
	f!(11, color=:red)
	# lines!(ax, Wildfires.cells(linestring1, 11), linewidth=1, color=:black)
	poly!(ax, marshall, alpha=.5)
	fig
end

# ╔═╡ a1b75d65-fe4a-4cbb-ac17-a5a04d3563e2
let 
	fig = Figure()
	ax = Axis(fig[1,1], title="Marshall Fire Polygon at H3 Resolutions 8-10")
	geom = GI.Polygon(marshall[1])
	function f!(i; kw...)
		c = Wildfires.cells(geom, i)
		poly!(ax, c; alpha = .3, kw...)
		lines!(ax, c; alpha=.3, linewidth=1, color=:black)
	end
	f!(8, color = :yellow)
	f!(9, color = :orange)
	f!(10, color = :red)
	fig
end

# ╔═╡ b9a101c8-4aa3-4b37-82a9-e2f12f6924d6
let 
	fig = Figure()
	ax = Axis(fig[1,1], title="Marshall Fire MultiPolygon at H3 Resolution 10")

	c = Wildfires.cells(marshall, 11)
	poly!(ax, c; color=:transparent, alpha=0.5)
	lines!(ax, c; linewidth=.5, color=:black)
	poly!(ax, marshall, alpha=.5)
	@warn "Some cells are missing...probably need our own implementation"
	fig
end

# ╔═╡ Cell order:
# ╟─11225dea-88f0-11f0-2be4-8733351417a6
# ╟─0728afb9-8c0b-4ad1-8e7c-ab672d751d65
# ╠═473b4ad2-5719-40df-a325-13bcf08061f0
# ╠═dbfd7610-620b-41ce-afa2-28b4bd359706
# ╟─60373761-5342-4f89-9507-44d09cc92ffd
# ╠═20b9c27e-3896-468b-9206-176da753b4cd
# ╟─6435f263-f671-4029-aa64-880ad596d453
# ╠═e88be00f-56ad-4a07-bb85-f2feb2bddb70
# ╟─a1b75d65-fe4a-4cbb-ac17-a5a04d3563e2
# ╠═b9a101c8-4aa3-4b37-82a9-e2f12f6924d6
