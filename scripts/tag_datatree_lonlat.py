"""
Script to add lon/lat coordinates to Sentinel-3 SLSTR datatree groups that lack them.
This version works with lazily loaded datatrees and can export metadata without loading data.

For lazily loaded datatrees:
- Coordinates are added lazily using assign_coords (doesn't load data arrays)
- Metadata can be exported without loading data using to_dict(data=False)
- For zarr output, coordinates are written directly to the store
"""
import os
import sys
import xarray as xr
import numpy as np
from typing import Optional, Tuple, Dict
import json
import pystac_client

# Add modules directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from modules.utils import get_datatree, locations
import pystac_client


def has_lonlat_coords(ds: Optional[xr.Dataset]) -> bool:
    """Check if dataset has latitude and longitude coordinates."""
    if ds is None:
        return False
    return 'latitude' in ds.coords and 'longitude' in ds.coords


def find_source_group_node(path: str, dt: xr.DataTree) -> Optional[xr.DataTree]:
    """
    Find the appropriate source group node for adding lon/lat coordinates.
    
    Returns the DataTree node that has lon/lat coordinates and matching dimensions.
    """
    # Special case: geometry_tn can use meteorology
    if path == "conditions/geometry_tn":
        try:
            return dt["conditions"]["meteorology"]
        except (KeyError, AttributeError):
            return None
    
    # Special case: geometry_to can use meteorology (same dimensions)
    if path == "conditions/geometry_to":
        try:
            return dt["conditions"]["meteorology"]
        except (KeyError, AttributeError):
            return None
    
    # Extract measurement group name from path
    parts = path.split("/")
    
    # Find measurement group name in path
    measurement_names = ["anadir", "aoblique", "bnadir", "boblique", 
                         "fnadir", "foblique", "inadir", "ioblique"]
    
    for name in measurement_names:
        if name in parts:
            try:
                return dt["measurements"][name]
            except (KeyError, AttributeError):
                continue
    
    # Handle time groups - map to corresponding measurement groups
    if path == "conditions/time_an":
        try:
            return dt["measurements"]["anadir"]
        except (KeyError, AttributeError):
            return None
    elif path == "conditions/time_bn":
        try:
            return dt["measurements"]["bnadir"]
        except (KeyError, AttributeError):
            return None
    elif path == "conditions/time_in":
        # Can use either fnadir or inadir (both have 1200 rows)
        try:
            return dt["measurements"]["inadir"]
        except (KeyError, AttributeError):
            try:
                return dt["measurements"]["fnadir"]
            except (KeyError, AttributeError):
                return None
    
    # Handle orphan pixels - use parent measurement group
    if "/orphan" in path:
        # Extract parent path: measurements/anadir/orphan -> measurements/anadir
        parent_path = path.rsplit("/orphan", 1)[0]
        try:
            parts = parent_path.split("/")
            node = dt
            for part in parts:
                node = node[part]
            return node
        except (KeyError, AttributeError):
            return None
    
    return None


def copy_coords_2d_lazy(target_ds: xr.Dataset, source_ds: xr.Dataset) -> xr.Dataset:
    """
    Copy lon/lat coordinates from source to target for 2D datasets (lazy version).
    
    This works with lazy datasets - coordinates are added without loading data arrays.
    """
    if not has_lonlat_coords(source_ds):
        return target_ds
    
    # Check dimension compatibility
    target_dims = set(target_ds.sizes.keys())
    source_dims = set(source_ds.sizes.keys())
    
    # Both should have rows and columns
    if not ({"rows", "columns"}.issubset(target_dims) and 
            {"rows", "columns"}.issubset(source_dims)):
        return target_ds
    
    # Check if dimensions match
    if (target_ds.sizes["rows"] != source_ds.sizes["rows"] or
        target_ds.sizes["columns"] != source_ds.sizes["columns"]):
        return target_ds
    
    # Extract coordinates (lazy - doesn't load data)
    lat = source_ds.coords["latitude"]
    lon = source_ds.coords["longitude"]
    
    # Copy coordinates (lazy operation) - use .data to extract the DataArray's data
    target_ds = target_ds.assign_coords(
        latitude=(("rows", "columns"), lat.data),
        longitude=(("rows", "columns"), lon.data)
    )
    
    # Copy attributes if present
    if hasattr(lat, 'attrs'):
        target_ds.coords["latitude"].attrs.update(lat.attrs)
    if hasattr(lon, 'attrs'):
        target_ds.coords["longitude"].attrs.update(lon.attrs)
    
    return target_ds


def copy_coords_1d_lazy(target_ds: xr.Dataset, source_ds: xr.Dataset) -> xr.Dataset:
    """
    Copy row-aligned lon/lat coordinates from source to target for 1D datasets (lazy version).
    
    For 1D arrays with 'rows' dimension, compute row-averaged coordinates lazily.
    """
    if not has_lonlat_coords(source_ds):
        return target_ds
    
    # Check if target has rows dimension
    if "rows" not in target_ds.sizes:
        return target_ds
    
    # Check if source has rows and columns
    if not ({"rows", "columns"}.issubset(source_ds.sizes)):
        return target_ds
    
    # Check if row counts match
    if target_ds.sizes["rows"] != source_ds.sizes["rows"]:
        return target_ds
    
    # Extract row-averaged coordinates (lazy operation)
    # For each row, compute mean lat/lon across columns
    lat_2d = source_ds.coords["latitude"]
    lon_2d = source_ds.coords["longitude"]
    
    # Compute mean along columns axis (axis=1) - lazy operation
    lat_1d = lat_2d.mean(dim="columns")
    lon_1d = lon_2d.mean(dim="columns")
    
    # Assign as 1D coordinates (lazy) - use .data to extract the DataArray's data
    target_ds = target_ds.assign_coords(
        latitude=("rows", lat_1d.data),
        longitude=("rows", lon_1d.data)
    )
    
    # Copy attributes if present
    if hasattr(source_ds.coords["latitude"], 'attrs'):
        target_ds.coords["latitude"].attrs.update(source_ds.coords["latitude"].attrs)
    if hasattr(source_ds.coords["longitude"], 'attrs'):
        target_ds.coords["longitude"].attrs.update(source_ds.coords["longitude"].attrs)
    
    return target_ds


def tag_datatree_lonlat_lazy(dt: xr.DataTree, verbose: bool = True) -> xr.DataTree:
    """
    Add lon/lat coordinates to all groups in the datatree that need them (lazy version).
    
    This function works with lazily loaded datatrees and doesn't load data arrays.
    It creates a new lazy datatree structure with coordinates added.
    
    Parameters
    ----------
    dt : xr.DataTree
        The lazily loaded datatree to tag
    verbose : bool
        Whether to print progress messages
    
    Returns
    -------
    xr.DataTree
        A new lazy datatree with lon/lat coordinates added to groups that needed them
    """
    if verbose:
        print("Tagging datatree groups with lon/lat coordinates (lazy mode)...")
        print("=" * 80)
    
    tagged_count = 0
    skipped_count = 0
    
    # Build dictionary of updated datasets (lazy)
    updated_groups = {}
    
    # Process all groups
    for path, node in dt.subtree_with_keys:
        # Handle root node
        if not path:
            if node.ds is not None:
                updated_groups[""] = node.ds
            continue
        
        # Check if it has a dataset
        if node.ds is None:
            continue
        
        target_ds = node.ds
        
        # Check if already has lon/lat coordinates
        if has_lonlat_coords(target_ds):
            updated_groups[path] = target_ds
            if verbose:
                print(f"  Skipping {path}: already has lon/lat coordinates")
            skipped_count += 1
            continue
        
        # Find source group node
        source_node = find_source_group_node(path, dt)
        if source_node is None or source_node.ds is None:
            updated_groups[path] = target_ds
            if verbose:
                print(f"  Skipping {path}: no source group found")
            skipped_count += 1
            continue
        
        source_ds = source_node.ds
        
        # Determine dimension pattern and copy coordinates accordingly
        target_dims = set(target_ds.sizes.keys())
        source_dims = set(source_ds.sizes.keys())
        
        tagged = False
        
        # Check for 2D spatial data (rows x columns)
        if {"rows", "columns"}.issubset(target_dims) and {"rows", "columns"}.issubset(source_dims):
            # Check if dimensions match
            if (target_ds.sizes["rows"] == source_ds.sizes["rows"] and
                target_ds.sizes["columns"] == source_ds.sizes["columns"]):
                target_ds = copy_coords_2d_lazy(target_ds, source_ds)
                tagged = True
                if verbose:
                    print(f"  Tagged {path} with 2D coordinates (lazy)")
        
        # Check for 1D spatial data (rows only)
        elif "rows" in target_dims and {"rows", "columns"}.issubset(source_dims):
            if target_ds.sizes["rows"] == source_ds.sizes["rows"]:
                target_ds = copy_coords_1d_lazy(target_ds, source_ds)
                tagged = True
                if verbose:
                    print(f"  Tagged {path} with 1D row-aligned coordinates (lazy)")
        
        # Check for orphan pixels - special handling needed
        elif "orphan_pixels" in target_dims:
            if verbose:
                print(f"  Skipping {path}: orphan pixels need special handling (not implemented)")
        
        # Non-spatial dimensions (detectors, uncertainties, etc.) - skip
        else:
            if verbose:
                print(f"  Skipping {path}: non-spatial dimensions")
        
        # Store the (possibly modified) dataset (still lazy)
        updated_groups[path] = target_ds
        
        if tagged:
            tagged_count += 1
        else:
            skipped_count += 1
    
    # Create new datatree from updated groups (lazy)
    dt_tagged = xr.DataTree.from_dict(updated_groups)
    
    # Copy root attributes if present
    if hasattr(dt, 'attrs'):
        dt_tagged.attrs.update(dt.attrs)
    
    if verbose:
        print()
        print("=" * 80)
        print(f"Tagging complete: {tagged_count} groups tagged, {skipped_count} groups skipped")
        print("Note: Datatree is still lazily loaded - data arrays not loaded into memory")
    
    return dt_tagged


def export_datatree_metadata(dt: xr.DataTree, output_path: str, verbose: bool = True):
    """
    Export datatree metadata (structure, coordinates, attributes) without loading data arrays.
    
    Parameters
    ----------
    dt : xr.DataTree
        The datatree to export metadata from
    output_path : str
        Path to save metadata JSON file
    verbose : bool
        Whether to print progress messages
    """
    if verbose:
        print(f"Exporting datatree metadata to {output_path}...")
    
    metadata_dict = {}
    
    for path, node in dt.subtree_with_keys:
        if node.ds is None:
            continue
        
        # Convert dataset to dict without data (only metadata, coordinates, structure)
        ds_dict = node.ds.to_dict(data=False)
        
        # Store with path as key
        key = path if path else "/"
        metadata_dict[key] = ds_dict
    
    # Add root attributes
    if hasattr(dt, 'attrs'):
        metadata_dict["_root_attrs"] = dt.attrs
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2, default=str)
    
    if verbose:
        print(f"Metadata exported successfully (no data arrays loaded)")


def get_datatree_with_id(location, day, collection='sentinel-3-slstr-l1-rbt', catalog='eopf'):
    """
    Get the datatree and item ID for a given location and day.
    Returns tuple (datatree, item_id).
    """
    if catalog == 'eopf':
        catalog_url = "https://stac.core.eopf.eodc.eu"
        stac_catalog = pystac_client.Client.open(catalog_url)
    else:
        raise ValueError(f"Invalid catalog: {catalog}")
    
    lon, lat = locations[location]
    
    items = list(stac_catalog.search(
        datetime=f"{day}/{day}",
        collections=[collection],
        intersects=dict(type="Point", coordinates=[lon, lat]),
    ).items())
    if not items:
        raise ValueError(f"No items found for location {location} on {day}")
    
    i = 0
    item = items[i]
    item_id = item.id
    
    dt = xr.open_datatree(item.assets["product"].href, **item.assets["product"].extra_fields["xarray:open_datatree_kwargs"])
    
    return dt, item_id


def main():
    """Main function to run the lazy tagging script."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description="Add lon/lat coordinates to Sentinel-3 SLSTR datatree groups (lazy mode)"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="biscay",
        choices=["biscay", "hel", "rysy", "svalbard"],
        help="Location to load datatree from STAC"
    )
    parser.add_argument(
        "--day",
        type=str,
        default="2025-06-17",
        help="Date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for tagged datatree (zarr format). If not specified, prints summary only."
    )
    parser.add_argument(
        "--metadata-only",
        type=str,
        nargs='?',
        const='auto',
        default=None,
        help="Export only metadata (structure, coordinates, attributes) to JSON file without loading data. If flag is used without a path, defaults to reports/metadata_{item_id}_{timestamp}.json"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Load datatree from STAC (lazy)
    print(f"Loading datatree from STAC catalog (lazy mode)...")
    print(f"Location: {args.location}, Day: {args.day}")
    dt, item_id = get_datatree_with_id(args.location, args.day)
    print(f'Opened the data tree with {len(dt.groups)} groups')
    
    # Tag with lon/lat coordinates (lazy)
    dt_tagged = tag_datatree_lonlat_lazy(dt, verbose=not args.quiet)
    
    # Export metadata - determine output path
    if args.metadata_only:
        if args.metadata_only == 'auto' or args.metadata_only is None:
            # Generate default filename with item ID and timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metadata_path = f"reports/metadata_{item_id}_{timestamp}.json"
        else:
            metadata_path = args.metadata_only
        
        # Ensure reports directory exists
        reports_dir = os.path.dirname(metadata_path) if os.path.dirname(metadata_path) else 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        export_datatree_metadata(dt_tagged, metadata_path, verbose=not args.quiet)
    
    # Save if output path specified
    if args.output:
        print(f"\nSaving tagged datatree to {args.output}...")
        print("Note: This will write coordinates to zarr. Data arrays remain lazy.")
        dt_tagged.to_zarr(args.output, mode="w")
        print("Done!")
    elif not args.metadata_only:
        print("\nNo output path specified. Use --output to save the tagged datatree.")
        print("Use --metadata-only to export just metadata without loading data.")


if __name__ == "__main__":
    main()

