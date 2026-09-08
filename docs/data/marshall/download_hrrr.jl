#--------------------------------------------------------------------------------# HRRR Wind Data Download for Marshall Fire
#
# Downloads 10m u/v wind and surface gust at 15-minute resolution from 17:00 UTC
# on 30 December to 02:00 UTC on 31 December 2021 from the NOAA HRRR archive on
# AWS. Byte-range requests pull only the needed
# GRIB2 messages, which are then reprojected to WGS84 and cropped.
#
# Run from the docs environment:
#   julia --project=docs docs/data/marshall/download_hrrr.jl
#
# Outputs:
#   wind_u_HHMM.tif, wind_v_HHMM.tif, wind_gust_HHMM.tif  (37 timestamps × 3 vars;
#   HHMM is UTC, 0000–0200 belong to the 31st)
#--------------------------------------------------------------------------------

using Downloads, ArchGDAL

const BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
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

# Time steps: (date, cycle_hour, source_file, fcst_label, output_suffix). Each
# hourly cycle supplies its analysis and the 15/30/45 min sub-hourly forecasts.
const CYCLES = [("20211230", h) for h in 17:23]
append!(CYCLES, [("20211231", h) for h in 0:2])
const TIMESTEPS = Tuple{String, Int, String, String, String}[]
for (date, hour) in CYCLES
    hh = lpad(hour, 2, '0')
    push!(TIMESTEPS, (date, hour, "wrfsfcf00", "anl", hh * "00"))
    hour == 2 && continue
    for minute in (15, 30, 45)
        push!(TIMESTEPS, (date, hour, "wrfsubhf01", "$minute min fcst", hh * string(minute)))
    end
end

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

    for (date, hour, source, fcst, suffix) in TIMESTEPS
        hh = lpad(hour, 2, '0')
        grib_url = "$BASE_URL/hrrr.$date/conus/hrrr.t$(hh)z.$(source).grib2"
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
            isfile(outpath) && continue
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
