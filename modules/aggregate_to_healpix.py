"""
Aggregate data to HEALPix cells using grouped mean.

This module implements the simplest aggregation method for converting
lon/lat gridded data to HEALPix format, based on the EOPF examples.

Key Functions:
- aggregate_to_healpix: Convert DataArray/Dataset to HEALPix format using grouped mean aggregation

References:
- EOPF_UTM_to_HEALPix.ipynb: `_to_healpix_cells_grouped_mean` function (lines 14531-14580)
"""

from healpix_geo import nested
import numpy as np
from typing import Optional, Union
import xarray as xr


def aggregate_to_healpix(
    data: Union[xr.DataArray, xr.Dataset],
    level: Optional[int] = None,
    ellipsoid: str = "WGS84",
    verbose: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Convert DataArray or Dataset to HEALPix format using grouped mean aggregation.
    
    This is the simplest aggregation method: maps each pixel to its HEALPix cell,
    then groups pixels by cell ID and computes the mean value for each cell.
    Multiple pixels mapping to the same HEALPix cell are averaged.
    
    Based on the EOPF examples:
    - EOPF_UTM_to_HEALPix.ipynb: `_to_healpix_cells_grouped_mean` function
    
    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input data with longitude and latitude coordinates.
        Must have spatial dimensions that can be stacked (e.g., 'rows', 'columns' or 'y', 'x').
    level : int, optional
        HEALPix refinement level. If None, will be inferred from existing 'healpix' coordinate
        or must be provided via 'crs' coordinate attributes.
    ellipsoid : str
        Ellipsoid for HEALPix calculation (default: 'WGS84')
    verbose : bool
        Whether to print progress information
    
    Returns
    -------
    xr.DataArray or xr.Dataset
        Aggregated data with HEALPix cell_ids as the primary dimension.
        Original spatial dimensions are replaced by 'cells' dimension.
        Each unique HEALPix cell contains the mean of all source pixels that mapped to it.
    
    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> 
    >>> # Create test DataArray with lon/lat coordinates
    >>> lon = np.linspace(-4.51, -4.49, 3)
    >>> lat = np.linspace(47.99, 48.01, 3)
    >>> lon_2d, lat_2d = np.meshgrid(lon, lat)
    >>> da = xr.DataArray(
    ...     np.random.rand(3, 3),
    ...     dims=['rows', 'columns'],
    ...     coords={'longitude': (['rows', 'columns'], lon_2d),
    ...             'latitude': (['rows', 'columns'], lat_2d)}
    ... )
    >>> 
    >>> # Aggregate to HEALPix
    >>> da_healpix = aggregate_to_healpix(da, level=12)
    >>> print(da_healpix)
    <xarray.DataArray (cells: N)>
    ...
    Coordinates:
        cell_ids  (cells) int64 ...
        crs       int8 ...
    """
    if nested is None:
        raise ImportError(
            "healpix_geo package is required. Install with: pip install healpix-geo"
        )
    
    # Check for required coordinates
    if "longitude" not in data.coords or "latitude" not in data.coords:
        raise ValueError(
            "Data must have 'longitude' and 'latitude' coordinates"
        )
    
    # Determine spatial dimensions
    lon = data.coords["longitude"]
    lat = data.coords["latitude"]
    
    # Get spatial dimension names
    if lon.ndim == 2:
        # 2D coordinates: use their dimensions
        spatial_dims = lon.dims
    elif lon.ndim == 1:
        # 1D coordinates: use their dimensions
        spatial_dims = lon.dims
    else:
        raise ValueError(
            f"Unsupported coordinate dimensions: lon.ndim={lon.ndim}, lat.ndim={lat.ndim}"
        )
    
    # Determine HEALPix level
    if level is None:
        # Try to get from existing healpix coordinate or crs metadata
        if "healpix" in data.coords:
            # Extract level from healpix coordinate attributes
            level = data.coords["healpix"].attrs.get("level")
            if level is None and "crs" in data.coords:
                level = data.coords["crs"].attrs.get("refinement_level")
        elif "crs" in data.coords:
            level = data.coords["crs"].attrs.get("refinement_level")
        
        if level is None:
            raise ValueError(
                "HEALPix level must be specified or present in 'healpix' or 'crs' coordinate attributes"
            )
    
    if verbose:
        print(f"Aggregating to HEALPix level {level}...")
    
    # Extract longitude and latitude values
    lon_vals = lon.values.ravel()
    lat_vals = lat.values.ravel()
    
    # Mask out NaN values
    valid_mask = ~(np.isnan(lon_vals) | np.isnan(lat_vals))
    
    # Calculate HEALPix cell IDs for all pixels
    # Initialize with NaN
    cell_ids_flat = np.full(len(lon_vals), np.nan, dtype=np.int64)
    
    if valid_mask.any():
        valid_lon = lon_vals[valid_mask]
        valid_lat = lat_vals[valid_mask]
        
        # Convert to HEALPix cell IDs using nested scheme
        cell_ids_valid = nested.lonlat_to_healpix(
            valid_lon, valid_lat, level, ellipsoid=ellipsoid
        )
        
        cell_ids_flat[valid_mask] = cell_ids_valid
    
    # Reshape to match original coordinate shape
    cell_ids = cell_ids_flat.reshape(lon.values.shape)
    
    # Stack spatial dimensions into 'cells' dimension
    # This flattens the spatial array while preserving data values
    if verbose:
        print(f"Stacking dimensions {spatial_dims} into 'cells'...")
    
    stacked = data.stack(cells=spatial_dims)
    
    # Attach HEALPix cell IDs as a coordinate on the 'cells' dimension
    stacked = stacked.assign_coords(cell_ids=("cells", cell_ids.ravel().astype("int64")))
    
    # Add CF-compliant metadata to cell_ids coordinate
    cell_ids_attrs = {
        "standard_name": "healpix",
        "units": "1",
        "long_name": "HEALPix cell ID",
        "level": level,
    }
    stacked["cell_ids"].attrs.update(cell_ids_attrs)
    
    # Save attributes before groupby (which may drop them)
    cell_ids_attrs_saved = dict(stacked["cell_ids"].attrs)
    
    # Group by cell_ids and compute mean for each group
    # Multiple pixels mapping to the same HEALPix cell are averaged
    if verbose:
        n_unique_before = len(np.unique(cell_ids_flat[valid_mask])) if valid_mask.any() else 0
        print(f"Grouping {len(cell_ids_flat[valid_mask])} pixels into {n_unique_before} unique cells...")
    
    # Group by cell_ids and take mean
    # After groupby, dimension is named 'cell_ids', rename dimension to 'cells'
    # but keep coordinate name as 'cell_ids'
    aggregated = stacked.groupby("cell_ids").mean()
    
    # Rename dimension from 'cell_ids' to 'cells', but keep coordinate name
    # We need to swap the dimension name while preserving the coordinate
    cell_ids_coord = aggregated.coords["cell_ids"]
    aggregated = aggregated.rename({"cell_ids": "cells"})
    # Restore cell_ids coordinate (rename changes both dim and coord name)
    aggregated = aggregated.assign_coords(cell_ids=("cells", cell_ids_coord.values))
    aggregated["cell_ids"].attrs.update(cell_ids_attrs_saved)
    
    # Add CF-compliant HEALPix metadata (crs coordinate) if not already present
    if "crs" not in aggregated.coords:
        crs_metadata = {
            "grid_mapping_name": "healpix",
            "refinement_level": level,
            "indexing_scheme": "nested",
            "reference_body": ellipsoid,
            "refinement_ratio": 4,
        }
        aggregated = aggregated.assign_coords(
            crs=xr.Variable((), np.int8(0), crs_metadata)
        )
    
    # Add grid_mapping and coordinates attributes
    if isinstance(aggregated, xr.DataArray):
        aggregated.attrs.update({
            "coordinates": "healpix",
            "grid_mapping": "crs",
        })
    else:
        # For Dataset, update attributes for each variable
        for var in aggregated.data_vars:
            if "cells" in aggregated[var].dims:
                aggregated[var].attrs.update({
                    "coordinates": "healpix",
                    "grid_mapping": "crs",
                })
    
    if verbose:
        n_cells = len(aggregated.coords["cell_ids"])
        print(f"Aggregation complete: {n_cells} unique HEALPix cells")
    
    return aggregated
