using Wildfires, Scratch, DataFrames, Rasters, ArchGDAL, Landfire

landfire_data = Scratch.get_scratch!(Wildfires, "landfire_data")
landfire_zip = joinpath(landfire_data, "landfire_products.zip")
landfire_products = filter!(Landfire.products(conus=true)) do p
    p.layer_name ∉ ("MF_F40FA24", "MF_FVCFA24", "MF_FVHFA24", "220ROADS_20")
end
landfire_products_df = DataFrame(landfire_products)

if !isfile(landfire_zip)
    Landfire.download(landfire_products, ext; output_projection="4326", dest=landfire_zip)
    run(`unzip -o $landfire_zip -d $landfire_data`)
end

landfire_file = filter(endswith(".tif"), readdir(landfire_data; join=true))[1]

landfire = Raster(landfire_file, checkmem = false)
landfire_bands = [landfire[Band=i].refdims[1][1] for i in 1:size(landfire, 3)]
landfire_dict = Dict(band => landfire[Band=i] for (i, band) in enumerate(landfire_bands))
