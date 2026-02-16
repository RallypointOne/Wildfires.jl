# Summary: Predicting Fire Propagation across Heterogeneous Landscapes Using WyoFire

**Authors:** Cory W. Ott, Bishrant Adhikari, Simon P. Alexander, Paddington Hodza, Chen Xu, Thomas A. Minckley

**Published:** Fire 2020, 3, 71 (December 2020)

## Overview

This paper evaluates the predictive performance of WyoFire, a Monte Carlo-driven wildfire simulation model developed at the University of Wyoming. The model was tested on 10 wildfires that occurred in Wyoming (and one in Montana) during the 2017 and 2019 fire seasons.

## Key Methodology

- **Monte Carlo Approach:** WyoFire uses Gaussian distributions for fuel moisture and meteorological data (relative humidity, temperature, wind direction, wind speed) to account for natural stochasticity in environmental conditions.
- **Fire Spread Model:** Based on Rothermel's equations combined with Wagner's crown fire model and Finney's elliptical propagation using the Huygens Wavelet principle.
- **Performance Metrics:** Area Difference Index (ADI), Precision, Recall, and F1 Score to compare simulated vs. observed fire perimeters.

## Data Sources

- HRRR meteorological forecasts (NOAA)
- VIIRS/MODIS active fire hotspots (NASA)
- Fuel moisture data (Wyoming State Forestry Division)
- Vegetation and fuels (LANDFIRE)
- Digital Elevation Models (USGS)
- Historical fire perimeters (GeoMAC)

## Key Findings

1. **Fuel loading is the dominant factor** affecting model performance, more than terrain complexity.

2. **Environment-dependent performance:**
   - Grassland/shrubland environments: Model tends to **over-predict** (ADIoe-ADIue > 32 for some fires)
   - Forested/woodland environments: Model tends to **under-predict** (ADIoe-ADIue < -2.5)
   - Mixed fuel types: Most balanced predictions

3. **Principal Components Analysis** identified three distinct groups of wildfire events based on fuel characteristics:
   - Group 1: Low fuel load, low fuel-bed continuity → higher under-prediction
   - Group 2: Medium-high fuel load, high fuel-bed continuity → higher over-prediction
   - Group 3: Mixed fuel types → most accurate predictions

4. **Optimal centroid distance (CD)** of 5m was identified across all fire types, though higher CD works better for grasslands and lower CD for forests.

## Conclusions

- WyoFire performs well across diverse landscapes when properly parameterized
- The model serves as an effective educational and risk assessment tool
- Monte Carlo simulation accounts for physical stochasticity inherent in natural wildfire behavior
- Future improvements can build on understanding of which environments cause over/under-prediction
