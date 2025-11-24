"""
Utility functions for working with Sentinel data.

"""
import pystac_client
import xarray as xr

locations = {
    "biscay":   (-4.50, 48.00),   # Bay of Biscay (France)
    "hel":      (18.81, 54.61),   # Hel (Baltic coast, Poland)
    "rysy":     (20.08, 49.18),   # Rysy (Tatra Mountains, Poland)
    "svalbard": (15.50, 78.22),   # Svalbard (Norway)
}


def get_datatree(location, day, collection='sentinel-3-slstr-l1-rbt', catalog='eopf'):
    """
    Get the datatree for a given location and day.
    """
    if catalog == 'eopf':
        catalog_url = "https://stac.core.eopf.eodc.eu"
        catalog = pystac_client.Client.open(catalog_url)
    else:
        raise ValueError(f"Invalid catalog: {catalog}")
    
    lon, lat = locations[location]

    print(f'Searching {catalog_url} for items in {collection} collection\n on {day} at lon={lon}, lat={lat}...')
    items = list(catalog.search(
        datetime=f"{day}/{day}",
        collections=[collection],
        intersects=dict(type="Point", coordinates=[lon, lat]),
    ).items())
    if not items:
        raise ValueError(f"No items found for location {location} on {day}")
    print(f'Found {len(items)} items')
    i = 0
    item = items[i]
    print(f'Opening the data tree for item no. {i+1}:\n {item.id}')
    dt = xr.open_datatree(item.assets["product"].href, **item.assets["product"].extra_fields["xarray:open_datatree_kwargs"])
    print(f'Opened the data tree with {len(dt.groups)} groups')
    return dt

def get_datatree_groups(dt: xr.DataTree) -> dict: # a bit useless
    """
    Get a nested dictionary of all groups and subgroups in a DataTree.
    
    Parameters
    ----------
    dt : xarray.DataTree
        The DataTree to traverse.
    
    Returns
    -------
    dict
        Nested dictionary where keys are group names and values are nested dicts
        for subgroups. Leaf groups have empty dicts as values.
    
    Examples
    --------
    >>> dt = xr.open_datatree("path/to/data.zarr")
    >>> groups = get_datatree_groups(dt)
    >>> # Result might be:
    >>> # {
    >>> #     "measurements": {
    >>> #         "anadir": {},
    >>> #         "aoblique": {}
    >>> #     },
    >>> #     "conditions": {
    >>> #         "geometry_tn": {},
    >>> #         "geometry_to": {}
    >>> #     }
    >>> # }
    """
    result = {}
    
    for path, node in dt.subtree_with_keys:
        # Skip root node (empty path)
        if not path:
            continue
        
        # Skip paths containing '.'
        if '.' in path:
            continue
        
        # Split path into components
        parts = path.split("/")
        
        # Build nested dictionary structure
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Add leaf node (empty dict for leaf groups)
        leaf_name = parts[-1]
        if leaf_name not in current:
            current[leaf_name] = {}
    
    return result

def parse_slstr_filename(filename, print_fields=True):
    """
    Parse the Sentinel-3 SLSTR L1 RBT filename and extract fields.

    Parameters
    ----------
    filename : str
        The product filename (with or without extension).

    Returns
    -------
    dict
        Parsed fields as a dictionary.
    
    Notes
    ----- 
    Sentinel-3 format:
    [0]=S3B, [1]=SL, [2]=1, [3]=RBT
    [4]=sensing start, [5]=sensing stop
    [6]=processing time
    [7]=orbit, [8]=cycle, [9]=rel_orbit, [10]=frame
    [11]=ESA, [12]=O, [13]=NT, [14]=version
    Allow for variable lengths
    """
    import re
    # Remove extension if present
    if filename.endswith(".SEN3") or filename.endswith(".nc") or filename.endswith(".zip"):
        filename = re.sub(r'\.[A-Za-z0-9]+$', '', filename)
    # Split on underscores, ignoring runs of multiple underscores
    parts = re.split(r'_+', filename)

    def safe(idx):
        return parts[idx] if idx < len(parts) else None
    # Parse datetimes as strings (YYYYMMDDTHHMMSS)
    fields = {
        "satellite": safe(0),    # S3A or S3B
        "instrument": safe(1),   # SL
        "processing_level": safe(2), # 1
        "product_type": safe(3), # RBT
        "sensing_start": safe(4),
        "sensing_stop": safe(5),
        "creation_date": safe(6),
        "orbit": safe(7),
        "cycle": safe(8),
        "relative_orbit": safe(9),
        "frame": safe(10),
        "agency": safe(11),
        "phase": safe(12),       # usually "O" (Operational)
        "timeliness": safe(13),  # e.g., "NT" (Near-Real-Time), "ST" (Short Time), "NT" (NRT)
        "version": safe(14)
    }
    # Optionally parse dates into datetime if needed
    from datetime import datetime
    fmt = "%Y%m%dT%H%M%S"
    for k in ["sensing_start", "sensing_stop", "creation_date"]:
        if fields[k]:
            try:
                fields[k + "_dt"] = datetime.strptime(fields[k], fmt)
            except Exception:
                fields[k + "_dt"] = None
    if print_fields:
        for key, value in fields.items():
            print(f"  {key}: {value}")
    return fields


def print_healpix_info(da: xr.DataArray):
    """
    Print information about HEALPix coordinates and metadata in a DataArray.
    
    Parameters
    ----------
    da : xr.DataArray
        DataArray with HEALPix coordinates (should have 'healpix' and optionally 'crs' coordinates)
    """
    print("\nTest DataArray with HEALPix coordinates:")
    print(f"  Coordinates: {list(da.coords.keys())}")
    
    # Get HEALPix level from crs coordinate (CF convention)
    if 'crs' in da.coords:
        crs_attrs = da.coords['crs'].attrs
        healpix_level = crs_attrs.get('refinement_level', 'unknown')
        indexing_scheme = crs_attrs.get('indexing_scheme', 'unknown')
        reference_body = crs_attrs.get('reference_body', 'unknown')
        print(f"  HEALPix level: {healpix_level}")
        print(f"  Indexing scheme: {indexing_scheme}")
        print(f"  Reference body: {reference_body}")
    else:
        print(f"  HEALPix level: unknown (crs coordinate not found)")
    
    if 'healpix' in da.coords:
        import numpy as np
        healpix_coord = da.coords['healpix']
        print(f"\nHEALPix cell IDs shape: {healpix_coord.shape}")
        
        # Count unique cells (excluding NaN)
        healpix_values = healpix_coord.values
        valid_mask = ~np.isnan(healpix_values)
        if valid_mask.any():
            unique_cells = len(np.unique(healpix_values[valid_mask]))
            print(f"Unique HEALPix cells: {unique_cells}")
        else:
            print(f"Unique HEALPix cells: 0 (all NaN)")
        
        print(f"\nHEALPix cell IDs:")
        print(healpix_values)
    else:
        print("\nWarning: 'healpix' coordinate not found in DataArray")
    
    # Show CRS metadata
    if 'crs' in da.coords:
        print(f"\nCRS metadata:")
        for key, value in da.coords['crs'].attrs.items():
            print(f"  {key}: {value}")


