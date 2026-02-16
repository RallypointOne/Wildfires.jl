# Summary: Developing a Geospatial Data-Driven Solution for Rapid Natural Wildfire Risk Assessment

**Authors:** Bishrant Adhikari, Chen Xu, Paddington Hodza, Thomas Minckley

**Published:** Applied Geography 126 (2021) 102382

## Overview

This paper presents a novel probabilistic wildfire risk assessment platform that automates the entire simulation workflow from data preparation to visualization. The system uses real-time wildfire observations and historical data to drive Monte Carlo-based simulations with Gaussian transformations of weather and fuel conditions.

## Key Innovations

1. **Gaussian Transformation:** Uses current observations as the mean and historical data to calculate variance, generating a distribution of possible environmental conditions between field observations.

2. **Fully Automated Workflow:** Synthesizes data preparation, analysis, and visualization into a single automated pipeline.

3. **Distributed Computing:** Deploys simulations on Apache Spark framework for parallel processing, dramatically reducing computation time.

## Technical Approach

- **Fire Spread Model:** Based on Rothermel's surface fire spread equation:
  ```
  R = (IRξ(1 + φw + φs)) / (ρbεQig)
  ```
- **Spatial Representation:** Uses elliptical boundaries with Huygens wavelet principle to model fire front propagation
- **Risk Mapping:** Overlays all Monte Carlo runs to create probability surface (more runs predicting an area = higher risk)

## Data Sources

| Dataset | Source | Spatial Resolution | Temporal Resolution |
|---------|--------|-------------------|---------------------|
| HRRR Weather | NOAA | 3 km | 1 hour |
| VIIRS Hotspots | NASA | 375 m | 12 hours |
| Dead Fuel Moisture | USFS | 2.5 km | 1 day |
| DEM | USGS | 10 m | Static |
| Vegetation/Fuels | LANDFIRE | 30 m | Jan 2014 |

## Computational Optimization

- **Apache Spark** with Hadoop Distributed File System (HDFS) for distributed storage
- **PostgreSQL/PostGIS** for spatial data retrieval
- **Threshold Strategy:** Uses traditional computing for < 1000 ignition points, switches to Apache Spark above that threshold
- **Results:** Computation time reduced from hours to minutes (628-1035% improvement)

## Validation Results

Three historical fires tested (Buffalo, Keystone, Pole Creek):

| Fire | F1 Score | Precision | Recall | ADI |
|------|----------|-----------|--------|-----|
| Buffalo | 0.96 | 0.67 | 0.97 | 0.53 |
| Keystone | 0.85 | 0.86 | 0.81 | 0.40 |
| Pole Creek | 0.85 | 0.63 | 0.92 | 0.69 |

- Model tends to over-predict (preferable for risk assessment)
- Under-prediction cases were minimal
- High recall values indicate good coverage of actual burned areas

## Key Conclusions

1. The Gaussian transformation effectively captures uncertainties in coarse resolution geographic data
2. Monte Carlo approach explicitly reflects real-world fire boundary uncertainties
3. Platform can provide **real-time decision-making support** for wildfire management
4. System is extensible to other regions by replacing the wildfire model's computational component

## Future Work

- Implement numerical weather prediction models to couple wildfire and atmosphere
- Integrate Ensemble Kalman Filter (EnKF) for parameter calibration based on new observations
- Explore alternative distribution functions beyond Gaussian
