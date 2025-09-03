module Wildfires

module Data
    using GeoJSON, Dates, DataFrames

    function occurrence_points()
        # url = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_FireOccurrenceAndPerimeter_01/MapServer/9/query?where=1=1&outFields=*&f=geojson"
        file = joinpath(@__DIR__, "..", "data", "National_USFS_Fire_Occurrence_Point_(Feature_Layer).geojson")
        GeoJSON.read(file)
    end

    function perimeters()

        # url = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_FireOccurrenceAndPerimeter_01/MapServer/11/query?where=1=1&outFields=*&f=geojson"
        # file = abspath(joinpath(@__DIR__, "..", "data", "perimeters.geojson"))
        # isfile(file) ? file : download(url, file)
        file = joinpath(@__DIR__, "..", "data", "National_USFS_Fire_Perimeter_(Feature_Layer).geojson")
        GeoJSON.read(file)
    end

    function marshall()
        file = joinpath(@__DIR__, "..", "data", "marshall.geojson")
        df = DataFrame(GeoJSON.read(file))
        df[!, :datetime] = Dates.unix2datetime.(df.poly_DateCurrent ./ 1000)
        sort!(df, :datetime)
        df
    end
end

end
