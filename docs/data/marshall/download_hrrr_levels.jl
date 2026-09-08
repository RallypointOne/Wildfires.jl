#--------------------------------------------------------------------------------# HRRR pressure-level analyses for the Marshall Fire domain
#
# Downloads the hourly HRRR analyses (wrfprsf00) from 17:00 UTC on 30 December
# to 02:00 UTC on 31 December 2021: geopotential height, temperature, specific
# humidity, and wind on the isobaric levels from 1000 to 300 hPa, plus surface
# pressure and terrain height. Byte-range requests pull only the needed GRIB2
# messages; the messages of one variable and hour are concatenated so GDAL warps
# them into one multi-band GeoTIFF (band k = LEVELS[k]).
#
# Run from the docs environment:
#   julia --project=docs docs/data/marshall/download_hrrr_levels.jl
#
# Outputs (37 files):
#   hrrr_{HGT,TMP,SPFH,UGRD,VGRD}_HHMM.tif   bands = pressure levels, hPa
#   hrrr_{PRES,HGT}_surface_HHMM.tif        single band
#--------------------------------------------------------------------------------

using Downloads, ArchGDAL

const BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
const OUTDIR = @__DIR__

# Target grid: WGS84, ~3km resolution, covering the Marshall Fire area with margin
const EXTENT = (-105.55, -104.85, 39.75, 40.15)
const RES = 0.03

const LEVELS = 1000:-25:300           # hPa; 300 hPa sits near 9 km, above the 8 km model top
const VARIABLES = ["HGT", "TMP", "SPFH", "UGRD", "VGRD"]
const CYCLES = [("20211230", h) for h in 17:23]
append!(CYCLES, [("20211231", h) for h in 0:2])

#--------------------------------------------------------------------------------# Helpers

function parse_idx(text)
    entries = []
    for line in split(strip(text), "\n")
        parts = split(line, ":")
        length(parts) >= 6 || continue
        push!(entries, (byte = parse(Int, parts[2]), var = parts[4], level = parts[5], fcst = strip(parts[6])))
    end
    entries
end

function byte_range(entries, var, level)
    idx = findfirst(e -> e.var == var && e.level == level && e.fcst == "anl", entries)
    idx === nothing && error("GRIB2 entry not found: $var:$level")
    stop = idx < length(entries) ? entries[idx + 1].byte - 1 : nothing
    return entries[idx].byte, stop
end

function download_messages(grib_url, entries, var, levels)
    tmpfile = tempname() * ".grib2"
    open(tmpfile, "w") do io
        for level in levels
            start, stop = byte_range(entries, var, level)
            range = stop === nothing ? "bytes=$start-" : "bytes=$start-$stop"
            Downloads.download(grib_url, io; headers = ["Range" => range])
        end
    end
    return tmpfile
end

function warp_to_geotiff(grib_path, output_path)
    ds = ArchGDAL.read(grib_path)
    ArchGDAL.gdalwarp([ds],
        ["-t_srs", "EPSG:4326",
         "-te", string(EXTENT[1]), string(EXTENT[3]), string(EXTENT[2]), string(EXTENT[4]),
         "-tr", string(RES), string(RES),
         "-r", "bilinear",
         "-of", "GTiff"]) do warped
        ArchGDAL.write(warped, output_path; driver = ArchGDAL.getdriver("GTiff"))
    end
end

function fetch!(grib_url, entries, var, levels, outpath)
    isfile(outpath) && return
    @info "Downloading" var output = basename(outpath)
    grib = download_messages(grib_url, entries, var, levels)
    try
        warp_to_geotiff(grib, outpath)
    finally
        rm(grib, force = true)
    end
end

#--------------------------------------------------------------------------------# Main

function main()
    for (date, hour) in CYCLES
        hh = lpad(hour, 2, '0')
        grib_url = "$BASE_URL/hrrr.$date/conus/hrrr.t$(hh)z.wrfprsf00.grib2"
        entries = parse_idx(String(take!(Downloads.download(grib_url * ".idx", IOBuffer()))))
        for var in VARIABLES
            fetch!(grib_url, entries, var, ["$p mb" for p in LEVELS], joinpath(OUTDIR, "hrrr_$(var)_$(hh)00.tif"))
        end
        fetch!(grib_url, entries, "PRES", ["surface"], joinpath(OUTDIR, "hrrr_PRES_surface_$(hh)00.tif"))
        fetch!(grib_url, entries, "HGT", ["surface"], joinpath(OUTDIR, "hrrr_HGT_surface_$(hh)00.tif"))
    end
    @info "Done"
end

main()
