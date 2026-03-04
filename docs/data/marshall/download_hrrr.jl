#--------------------------------------------------------------------------------# HRRR Wind Data Download for Marshall Fire
#
# Downloads 10m u/v wind and surface gust at 15-minute resolution (17:00–19:00 UTC)
# from the NOAA HRRR archive on AWS. Byte-range requests pull only the needed
# GRIB2 messages, which are then reprojected to WGS84 and cropped.
#
# Run from the docs environment:
#   julia --project=docs docs/data/marshall/download_hrrr.jl
#
# Outputs:
#   wind_u_HHMM.tif, wind_v_HHMM.tif, wind_gust_HHMM.tif  (9 timestamps × 3 vars)
#--------------------------------------------------------------------------------

using Downloads, ArchGDAL

const BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20211230/conus"
const OUTDIR = @__DIR__

# Target grid: WGS84, ~3km resolution, covering Marshall Fire area with margin
const EXTENT = (-105.55, -104.85, 39.75, 40.15)
const RES = 0.03

# Variables: (GRIB2 name, GRIB2 level, output prefix)
const VARIABLES = [
    ("UGRD", "10 m above ground", "wind_u"),
    ("VGRD", "10 m above ground", "wind_v"),
    ("GUST", "surface",           "wind_gust"),
]

# Time steps: (cycle_hour, source_file, fcst_label, output_suffix)
const TIMESTEPS = [
    (17, "wrfsfcf00",  "anl",         "1700"),
    (17, "wrfsubhf01", "15 min fcst", "1715"),
    (17, "wrfsubhf01", "30 min fcst", "1730"),
    (17, "wrfsubhf01", "45 min fcst", "1745"),
    (18, "wrfsfcf00",  "anl",         "1800"),
    (18, "wrfsubhf01", "15 min fcst", "1815"),
    (18, "wrfsubhf01", "30 min fcst", "1830"),
    (18, "wrfsubhf01", "45 min fcst", "1845"),
    (19, "wrfsfcf00",  "anl",         "1900"),
]

#--------------------------------------------------------------------------------# Helpers

function parse_idx(text)
    entries = []
    for line in split(strip(text), "\n")
        parts = split(line, ":")
        length(parts) >= 6 || continue
        push!(entries, (
            num   = parse(Int, parts[1]),
            byte  = parse(Int, parts[2]),
            var   = parts[4],
            level = parts[5],
            fcst  = strip(parts[6]),
        ))
    end
    entries
end

function find_byte_range(entries, var, level, fcst)
    idx = findfirst(e -> e.var == var && e.level == level && e.fcst == fcst, entries)
    idx === nothing && error("GRIB2 entry not found: $var:$level:$fcst")
    start_byte = entries[idx].byte
    end_byte = idx < length(entries) ? entries[idx + 1].byte - 1 : nothing
    return start_byte, end_byte
end

function download_grib_message(grib_url, start_byte, end_byte)
    tmpfile = tempname() * ".grib2"
    range = end_byte === nothing ? "bytes=$start_byte-" : "bytes=$start_byte-$end_byte"
    Downloads.download(grib_url, tmpfile; headers=["Range" => range])
    return tmpfile
end

function warp_to_geotiff(grib_path, output_path)
    ds = ArchGDAL.read(grib_path)
    ArchGDAL.gdalwarp(
        [ds],
        ["-t_srs", "EPSG:4326",
         "-te", string(EXTENT[1]), string(EXTENT[3]), string(EXTENT[2]), string(EXTENT[4]),
         "-tr", string(RES), string(RES),
         "-r", "bilinear",
         "-of", "GTiff"],
    ) do warped_ds
        ArchGDAL.write(warped_ds, output_path; driver=ArchGDAL.getdriver("GTiff"))
    end
end

#--------------------------------------------------------------------------------# Main

function main()
    idx_cache = Dict{String, Vector}()

    for (hour, source, fcst, suffix) in TIMESTEPS
        hh = lpad(hour, 2, '0')
        grib_url = "$BASE_URL/hrrr.t$(hh)z.$(source).grib2"
        idx_url  = "$grib_url.idx"

        # Cache idx files (one per unique GRIB2 file)
        if !haskey(idx_cache, idx_url)
            @info "Fetching index" idx_url
            idx_text = String(take!(Downloads.download(idx_url, IOBuffer())))
            idx_cache[idx_url] = parse_idx(idx_text)
        end
        entries = idx_cache[idx_url]

        for (var, level, prefix) in VARIABLES
            outpath = joinpath(OUTDIR, "$(prefix)_$(suffix).tif")
            @info "Downloading" var level fcst output=basename(outpath)

            start_byte, end_byte = find_byte_range(entries, var, level, fcst)
            grib_path = download_grib_message(grib_url, start_byte, end_byte)
            try
                warp_to_geotiff(grib_path, outpath)
            finally
                rm(grib_path, force=true)
            end
        end
    end

    @info "Done! Downloaded $(length(TIMESTEPS) * length(VARIABLES)) files to $OUTDIR"
end

main()
