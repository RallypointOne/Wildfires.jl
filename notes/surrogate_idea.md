# Idea: Fit Differentiable Surrogates Over Raster Layers

- Context: In wildfire models, quantities like slope, aspect, or fuel moisture often appear in PDE terms (Rothermal equation):

$$
\frac{\partial\phi}{\partial t} + R(\theta, w, s, f)|\nabla\phi| = 0
$$

- Problem:

  - Raster data are discrete and often discontinuous (nearest-neighbor or bilinear). Finite differences on raw rasters yield noisy, non-smooth gradients.

## Main Idea

- Fit smooth surrogates $f(t, x, y | \theta)$ for geospatial layers.

> Static datasets will be $f(x,y|\theta)$

#### Results:
- Fully differentiable pipeline
- Expanding idea to vector/mesh data is trivial:
  - Less "glue" required to join different types of data: raster, vector, mesh, etc.
  - Our proposal mentions "data fusion"
- Parameters $\theta$ for each layer are learnable: Can assimilate sensor data with raster data via gradient-based optimization (ADAM, LBFGS, etc.).
  - The “environment” itself becomes a tunable, learnable part of the digital twin.
- Faster performance:
  - Evaluating compiled Julia functions rather than disk-backed raster reads.
- Reduces problems from:
  - Numerical instability: No finite differencing.
  - Aligning rasters data with PDE meshes or level set data.
- Computable "sensitivites": How output changes w.r.t how a given input changes, e.g. how does rate of spread change as fuel moisture content changes?

## Existing/Related Work:

- Physics-Informed Neural Operators (PINOs).
  - https://github.com/SciML/NeuralOperators.jl
    - Uses Reactant.jl

The field of neural fields / implicit neural representations (INRs / coordinate-based networks): These map spatial (and sometimes temporal) coordinates directly to a continuous output (e.g., height, density, color). These networks are inherently differentiable in the input coordinates.


- For example: Neural Elevation Models for Terrain Mapping and Path Planning (“NEMos”) introduce a continuous, differentiable height-field representation of terrain derived from imagery, rather than a traditional discrete DEM.

- Another: Spatial Implicit Neural Representations for Global‑Scale Species Mapping uses implicit neural representations for large-scale geospatial mapping tasks.


- In the geophysics domain: Implicit neural representation for potential field geophysics uses a coordinate-MLP to represent potential (gravity/magnetic) fields from survey data, including ability to compute gradients via AD.

- Terrain-specific continuous surface modeling: ImplicitTerrain: a Continuous Surface Model for Terrain Data Analysis (2024) presents a pipeline for high-resolution terrain data using INRs, enabling topological and derivative analyses on the continuous model.

- In computer graphics/rendering, the notion of differentiable rasterization / rendering is well developed (though aimed at graphics rather than geoscience). E.g., Differentiable Vector Graphics Rasterization for Editing and Learning (2020) introduces a differentiable renderer from vector primitives to raster images.

## Opportunity

- While INRs for terrain and geospatial fields exist, their use for environmental layers like fuel moisture, dynamic vegetation/fuel load, or wildfire-specific inputs seems less mature.

- Many geospatial INRs focus on static fields (terrain height, species presence) rather than rapidly evolving fields or coupling to PDEs (fire spread, fuel consumption).

> ChatGPT didn’t locate examples of raster → differentiable surrogate layer specifically for wildfire spread models (fuel moisture, terrain closure for fire PDEs) in the literature. Thus there is a novel opportunity to apply this concept explicitly in the wildfire digital twin context.

- Technique established in geoscience but not yet applied to wildfire.
  - “to our knowledge, first application of differentiable surrogate raster layers for fuel-moisture/terrain used in a wildfire modeling”.
