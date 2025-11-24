"""
Converting lon/lat coordinates to HEALPix cell IDs.

Key Functions:
- calculate_pixel_sizes: Calculate pixel sizes from lon/lat coordinates as distance between adjacent pixels in row and column directions
- select_healpix_level: Select HEALPix level based on pixel size
- add_healpix_metadata: Add CF-compliant HEALPix metadata to a DataArray
- add_healpix_to_dataarray: Add HEALPix cell IDs to a DataArray based on lon/lat coordinates
"""
from healpix_geo import nested
from pyproj import Geod # for calculating pixel sizes
import numpy as np
from typing import Optional, Tuple
import xarray as xr

# HEALPix level edge lengths dictionary
#   level: edge_length (meters)
# From healpix-geo documentation: 
# https://healpix-geo.readthedocs.io/en/latest/healpix/levels.html.
HEALPIX_EDGE_LENGTHS = {
    0: 6519662.5680,
    1: 3259831.2840,
    2: 1629915.6420,
    3: 814957.8210,
    4: 407478.9105,
    5: 203739.4552,
    6: 101869.7276,
    7: 50934.8638,
    8: 25467.4319,
    9: 12733.7160,
    10: 6366.8580,
    11: 3183.4290,
    12: 1591.7145,
    13: 795.8572,
    14: 397.9286,
    15: 198.9643,
    16: 99.4822,
    17: 49.7411,
    18: 24.8705,
    19: 12.4353,
    20: 6.2176,
    21: 3.1088,
    22: 1.5544,
    23: 0.7772,
    24: 0.3886,
    25: 0.1943,
    26: 0.0972,
    27: 0.0486,
    28: 0.0243,
    29: 0.0121,
}

def calculate_pixel_sizes(da: xr.DataArray, verbose: bool = False) -> Tuple[float, float]:
    """
    Calculate pixel sizes from lon/lat coordinates in a DataArray
    as distance between adjacent pixels in row and column directions
    on a ellipsoid, using pyproj.Geod.inv.
    
    Parameters
    ----------
    da : xr.DataArray
        DataArray with longitude and latitude coordinates
    verbose : bool
        Whether to print pixel size statistics
    
    Returns
    -------
    Tuple[float, float]
        Tuple of (mean_row_size, mean_col_size) in meters
    """
    # Check for coordinates
    if 'longitude' not in da.coords or 'latitude' not in da.coords:
        raise ValueError("DataArray must have 'longitude' and 'latitude' coordinates")
    
    lon = da.coords['longitude'].values
    lat = da.coords['latitude'].values
    
    # Initialize Geod object for ellipsoidal distance calculations (WGS84)
    geod = Geod(ellps='WGS84')
    
    # Calculate row-wise distances (vertical/north-south direction)
    # Distance between pixel (i, j) and pixel (i+1, j)
    if lon.ndim == 2 and lat.ndim == 2:
        # For 2D longitude/latitude coordinates:
        # Compute the coordinates of each pair of vertically adjacent pixels (rows).
        # - (lon1_rows, lat1_rows): all points except the last row
        # - (lon2_rows, lat2_rows): all points except the first row
        # The .ravel() flattens arrays for vectorised distance computation.
        lon1_rows = lon[:-1, :].ravel()   # longitude of row i, for all i except last
        lat1_rows = lat[:-1, :].ravel()   # latitude  of row i, for all i except last
        lon2_rows = lon[1:,  :].ravel()   # longitude of row i+1, paired with row i
        lat2_rows = lat[1:,  :].ravel()   # latitude  of row i+1, paired with row i
        
        # Column-wise distances (horizontal/east-west direction)
        lon1_cols = lon[:, :-1].ravel()
        lat1_cols = lat[:, :-1].ravel()
        lon2_cols = lon[:, 1:].ravel()
        lat2_cols = lat[:, 1:].ravel()
    elif lon.ndim == 1 and lat.ndim == 1:
        # 1D case - only row distances
        lon1_rows = lon[:-1]
        lat1_rows = lat[:-1]
        lon2_rows = lon[1:]
        lat2_rows = lat[1:]
        lon1_cols = np.array([])
        lat1_cols = np.array([])
        lon2_cols = np.array([])
        lat2_cols = np.array([])
    else:
        raise ValueError(f"Unsupported coordinate dimensions: lon.shape={lon.shape}, lat.shape={lat.shape}")
    
    # Mask out NaN values for rows
    valid_rows = ~(np.isnan(lon1_rows) | np.isnan(lat1_rows) | 
                   np.isnan(lon2_rows) | np.isnan(lat2_rows))
    
    if valid_rows.any():
        _, _, row_dists = geod.inv(
            lon1_rows[valid_rows], lat1_rows[valid_rows],
            lon2_rows[valid_rows], lat2_rows[valid_rows]
        )
        row_distances = np.abs(row_dists)  # Use absolute values
    else:
        row_distances = np.array([])
    
    # Mask out NaN values for columns
    if len(lon1_cols) > 0:
        valid_cols = ~(np.isnan(lon1_cols) | np.isnan(lat1_cols) | 
                       np.isnan(lon2_cols) | np.isnan(lat2_cols))
        
        if valid_cols.any():
            _, _, col_dists = geod.inv(
                lon1_cols[valid_cols], lat1_cols[valid_cols],
                lon2_cols[valid_cols], lat2_cols[valid_cols]
            )
            col_distances = np.abs(col_dists)  # Use absolute values
        else:
            col_distances = np.array([])
    else:
        col_distances = np.array([])
    
    # Calculate mean pixel sizes
    mean_row_size = np.mean(row_distances) if len(row_distances) > 0 else 0.0
    mean_col_size = np.mean(col_distances) if len(col_distances) > 0 else mean_row_size  # Use row size if no columns
    
    if verbose:
        print("=" * 60)
        print("PIXEL SIZE STATISTICS")
        print("=" * 60)
        
        if len(row_distances) > 0:
            print(f"\nRow-wise pixel sizes (North-South direction):")
            print(f"  Mean:   {mean_row_size:.2f} m")
            print(f"  Median: {np.median(row_distances):.2f} m")
            print(f"  Min:    {np.min(row_distances):.2f} m")
            print(f"  Max:    {np.max(row_distances):.2f} m")
        
        if len(col_distances) > 0:
            print(f"\nColumn-wise pixel sizes (East-West direction):")
            print(f"  Mean:   {mean_col_size:.2f} m")
            print(f"  Median: {np.median(col_distances):.2f} m")
            print(f"  Min:    {np.min(col_distances):.2f} m")
            print(f"  Max:    {np.max(col_distances):.2f} m")
    
    return mean_row_size, mean_col_size


def select_healpix_level(pixel_size: float, verbose: bool = False) -> int:
    """
    Select HEALPix level based on pixel size.
    
    Selects the smallest level where edge_length > pixel_size.
    Iterates from largest (level 0) to smallest (level 29) to find
    the smallest level where edge_length > pixel_size.
    
    Parameters
    ----------
    pixel_size : float
        Mean pixel size in meters
    verbose : bool
        Whether to print selection information
    
    Returns
    -------
    int
        HEALPix level (depth)
    """
    # Find the smallest level where edge_length > pixel_size
    # Iterate from largest to smallest level (level 0 to 29)
    # and find the first level where edge_length <= pixel_size,
    # then return the previous level
    selected_level = 0  # Default to level 0
    
    for level in sorted(HEALPIX_EDGE_LENGTHS.keys()):
        edge_length = HEALPIX_EDGE_LENGTHS[level]
        if edge_length > pixel_size:
            selected_level = level
        else:
            # Once we find a level where edge_length <= pixel_size, stop
            # The previous level (selected_level) is the answer
            break
    
    if verbose:
        edge_length = HEALPIX_EDGE_LENGTHS[selected_level]
        print(f"\nSelected HEALPix level: {selected_level}")
        print(f"  Pixel size: {pixel_size:.2f} m")
        print(f"  Edge length: {edge_length:.2f} m")
        print(f"  Ratio: {edge_length / pixel_size:.2f}x")
    
    return selected_level


def add_healpix_metadata(da: xr.DataArray, depth: int, ellipsoid: str = 'WGS84') -> xr.DataArray:
    """
    Add CF-compliant HEALPix metadata to a DataArray with healpix coordinate.
    CF stands for Climate and Forecast metadata conventions.
    https://cfconventions.org/
    
    Parameters
    ----------
    da : xr.DataArray
        DataArray with 'healpix' coordinate already added
    depth : int
        HEALPix depth/level
    ellipsoid : str
        Ellipsoid for HEALPix calculation (default: 'WGS84')
    
    Returns
    -------
    xr.DataArray
        DataArray with CF-compliant HEALPix metadata added
    """
    if 'healpix' not in da.coords:
        raise ValueError("DataArray must have 'healpix' coordinate before adding metadata")
    
    # 1. Add grid mapping coordinate (crs)
    crs_metadata = {
        'grid_mapping_name': 'healpix',
        'refinement_level': depth,
        'indexing_scheme': 'nested',
        'reference_body': ellipsoid,
        'refinement_ratio': 4,
    }
    da_with_metadata = da.assign_coords(
        crs=xr.Variable((), np.int8(0), crs_metadata)
    )
    
    # 2. Update healpix coordinate attributes (CF convention)
    da_with_metadata.coords['healpix'].attrs.update({
        'standard_name': 'healpix',
        'units': '1',
        'long_name': 'HEALPix cell ID',
    })
    
    # 3. Add grid_mapping and coordinates attributes to the DataArray itself
    da_with_metadata.attrs.update({
        'coordinates': 'healpix',
        'grid_mapping': 'crs',
    })
    
    return da_with_metadata


def add_healpix_to_dataarray(da: xr.DataArray, depth: Optional[int] = None, 
                             ellipsoid: str = 'WGS84', verbose: bool = False) -> xr.DataArray:
    """
    Add HEALPix cell IDs to a DataArray based on lon/lat coordinates.
    
    Parameters
    ----------
    da : xr.DataArray
        DataArray with longitude and latitude coordinates
    depth : int, optional
        HEALPix depth/level. If None, automatically selected based on pixel size.
    ellipsoid : str
        Ellipsoid for HEALPix calculation (default: 'WGS84')
    verbose : bool
        Whether to print progress information
    
    Returns
    -------
    xr.DataArray
        DataArray with 'healpix' coordinate added
    """
    if nested is None:
        raise ImportError("healpix_geo package is required. Install with: pip install healpix-geo")
    
    # Check for coordinates
    if 'longitude' not in da.coords or 'latitude' not in da.coords:
        raise ValueError("DataArray must have 'longitude' and 'latitude' coordinates")
    
    lon = da.coords['longitude']
    lat = da.coords['latitude']
    
    # Calculate pixel sizes if depth not specified
    if depth is None:
        if verbose:
            print("Calculating pixel sizes...")
        mean_row_size, mean_col_size = calculate_pixel_sizes(da, verbose=verbose)
        
        # Use the larger of row/column sizes to ensure coverage
        pixel_size = max(mean_row_size, mean_col_size)
        
        if verbose:
            print(f"\nUsing pixel size: {pixel_size:.2f} m (max of row/column)")
        
        # Select appropriate HEALPix level
        depth = select_healpix_level(pixel_size, verbose=verbose)
    else:
        if verbose:
            edge_length = HEALPIX_EDGE_LENGTHS.get(depth, "unknown")
            print(f"Using specified HEALPix level: {depth} (edge length: {edge_length} m)")
    
    # Get coordinate values (compute if lazy)
    if hasattr(lon.data, 'compute'):
        lon_vals = lon.compute().values
        lat_vals = lat.compute().values
    else:
        lon_vals = lon.values
        lat_vals = lat.values
    
    # Flatten coordinates for HEALPix calculation
    lon_flat = lon_vals.ravel()
    lat_flat = lat_vals.ravel()
    
    # Mask out NaN values
    valid_mask = ~(np.isnan(lon_flat) | np.isnan(lat_flat))
    
    # Calculate HEALPix cell IDs
    if verbose:
        print(f"\nCalculating HEALPix cell IDs at level {depth}...")
    
    # Initialize cell IDs array with NaN
    cell_ids_flat = np.full(len(lon_flat), np.nan, dtype=np.int64)
    
    if valid_mask.any():
        # Calculate HEALPix IDs for valid pixels
        valid_lon = lon_flat[valid_mask]
        valid_lat = lat_flat[valid_mask]
        
        cell_ids_valid = nested.lonlat_to_healpix(
            valid_lon, valid_lat, depth, ellipsoid=ellipsoid
        )
        
        cell_ids_flat[valid_mask] = cell_ids_valid
    
    # Reshape to match original coordinate shape
    cell_ids = cell_ids_flat.reshape(lon_vals.shape)
    
    
    # Determine coordinate dimensions based on lon/lat dimensions
    if lon.ndim == 2:
        coord_dims = lon.dims
    elif lon.ndim == 1:
        coord_dims = lon.dims
    else:
        coord_dims = da.dims[:lon.ndim]  # Fallback
    
    # Add as coordinate to DataArray
    da_with_healpix = da.assign_coords(
        healpix=(coord_dims, cell_ids)
    )
    
    # Add CF-compliant HEALPix metadata
    da_with_healpix = add_healpix_metadata(da_with_healpix, depth=depth, ellipsoid=ellipsoid)
    
    if verbose:
        n_valid = np.sum(valid_mask)
        n_total = len(lon_flat)
        n_unique = len(np.unique(cell_ids_valid)) if valid_mask.any() else 0
        print(f"Added HEALPix coordinates:")
        print(f"  Valid pixels: {n_valid} / {n_total}")
        print(f"  Unique cells: {n_unique}")
        print(f"  Level: {depth}")
        print(f"  Grid mapping: crs (CF-compliant)")
    
    return da_with_healpix
