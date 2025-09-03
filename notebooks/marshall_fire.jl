### A Pluto.jl notebook ###
# v0.20.13

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

# ╔═╡ 11225dea-88f0-11f0-2be4-8733351417a6
begin 
	using Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
	using Wildfires, Plots, PlutoUI, DataFrames
	PlutoUI.TableOfContents()
end

# ╔═╡ 0728afb9-8c0b-4ad1-8e7c-ab672d751d65
md"""
# Load DataFrame
"""

# ╔═╡ baf8fb24-0ddc-47f7-8ea6-a6ae86d212e5
df = Wildfires.Data.marshall()

# ╔═╡ c26ce79a-15df-40ba-a8ea-e359d685b6a6
@bind i Slider(1:nrow(df), show_value=true)

# ╔═╡ aef45998-d69a-4fc6-b3ae-9ddc54257a08
df.datetime[i]

# ╔═╡ a5cdc3f0-8b6f-4ab0-867f-230d3264d772
plot(df.geometry[i])

# ╔═╡ a23409ee-114a-4cda-8c68-a4b217ed5afe
plot(df.geometry[1:i])

# ╔═╡ Cell order:
# ╟─11225dea-88f0-11f0-2be4-8733351417a6
# ╟─0728afb9-8c0b-4ad1-8e7c-ab672d751d65
# ╠═baf8fb24-0ddc-47f7-8ea6-a6ae86d212e5
# ╠═c26ce79a-15df-40ba-a8ea-e359d685b6a6
# ╠═aef45998-d69a-4fc6-b3ae-9ddc54257a08
# ╠═a5cdc3f0-8b6f-4ab0-867f-230d3264d772
# ╠═a23409ee-114a-4cda-8c68-a4b217ed5afe
