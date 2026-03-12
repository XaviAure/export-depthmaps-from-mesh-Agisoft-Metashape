# SPDX-License-Identifier: GPL-3.0-or-later
"""
High-resolution depth map exporter for Agisoft Metashape.

Exports 16-bit depth maps, object masks, and metadata for each camera in a
Metashape project. Depth values are corrected to real-world scale so they
can be used directly in downstream workflows such as depth fusion with
RTI/photometric stereo or CNC milling of tactile reproductions.

License: GNU General Public License v3.0 or later (GPL-3.0-or-later)
"""

import Metashape
import numpy as np
import imageio
import os
import time
import tempfile
import json
from scipy.ndimage import generic_filter, binary_erosion
from scipy.interpolate import griddata

# ============================================================================
# CONFIGURATION
# ============================================================================

output_folder = Metashape.app.getExistingDirectory("Select output folder for depth maps")
if not output_folder:
    raise RuntimeError("No output folder selected. Aborting.")

MAX_INPAINTING_ITERATIONS = 8  # Quality default for filling interior holes
FILL_VALUE = 65535
# Use full range to avoid trimming true geometric extremes.
PERCENTILE_LOW = 0.0
PERCENTILE_HIGH = 100.0
# Reserve 65535 strictly as NoData/background.
MAX_VALID_DEPTH_VALUE = 65534

# Edge protection: don't inpaint this many pixels from the edge
# Only used when inpainting is enabled - set to 0 when skipping inpainting
EDGE_PROTECTION_PIXELS = 5

# Set to False for quality exports (fills interior holes), True for fast preview exports.
SKIP_INPAINTING = False

# How depth is measured from the camera. "camera_z" (perpendicular distance)
# is the standard for Metashape mesh exports.
DEPTH_METRIC_MODEL = "camera_z"  # Options: camera_z, ray_distance

# When skipping inpainting, no edge erosion is needed (it only causes border artifacts)
if SKIP_INPAINTING:
    EDGE_PROTECTION_PIXELS = 0

# ============================================================================
# INITIALIZATION
# ============================================================================

print("=" * 80)
print("METASHAPE DEPTH MAP EXPORTER (INITIAL RELEASE, GPL-3.0)")
print("=" * 80)
print(f"Output folder: {output_folder}")
print(f"Inpainting: {'DISABLED (fast mode)' if SKIP_INPAINTING else f'ENABLED ({MAX_INPAINTING_ITERATIONS} iterations)'}")
print()

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("✓ Output folder created")
else:
    print("✓ Output folder exists")

doc = Metashape.app.document
if not doc.chunk:
    raise RuntimeError("No active chunk found.")
chunk = doc.chunk
print("✓ Active chunk loaded")

# Extract the project's scale factor to convert depth values to real-world units
chunk_transform = chunk.transform.matrix
chunk_scale = (chunk_transform[0, 0]**2 + chunk_transform[0, 1]**2 + chunk_transform[0, 2]**2)**0.5
if not np.isfinite(chunk_scale) or chunk_scale <= 0:
    raise RuntimeError(f"Invalid chunk transform scale: {chunk_scale}")
print(f"✓ Chunk transform scale detected: {chunk_scale:.6f}")
print(f"  (Will multiply depth values by this to get world metric depths)")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_depth_channel(depth_map):
    """Reads the depth values from a Metashape depth image."""
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
        temp_filename = tmp_file.name
    
    try:
        depth_map.save(temp_filename)
        depth_image = imageio.imread(temp_filename)
        
        if depth_image.ndim == 3:
            depth_channel = depth_image[:, :, 0]
        elif depth_image.ndim == 2:
            depth_channel = depth_image
        else:
            raise ValueError(f"Unexpected depth map dimensions: {depth_image.shape}")
    finally:
        try:
            os.remove(temp_filename)
        except OSError:
            pass
    
    return depth_channel


def create_object_mask(depth_channel, edge_erosion=EDGE_PROTECTION_PIXELS):
    """
    Create a mask showing where the object is in the depth map.
    Optionally shrinks the mask edges to avoid artefacts during hole filling.

    Args:
        depth_channel: Raw depth values from Metashape
        edge_erosion: Number of pixels to shrink from object edges

    Returns:
        Mask image (True = object, False = background)
    """
    # Valid object pixels are those with depth > 0
    object_mask = depth_channel > 0
    
    # Erode the mask to protect edges
    if edge_erosion > 0:
        from scipy.ndimage import binary_erosion
        structure = np.ones((edge_erosion * 2 + 1, edge_erosion * 2 + 1))
        object_mask = binary_erosion(object_mask, structure=structure)
        print(f"  Object mask eroded by {edge_erosion} pixels to protect edges")
    
    return object_mask


def get_camera_intrinsics(calibration):
    """Returns camera calibration parameters (focal lengths and principal point)."""
    f = float(getattr(calibration, 'f', 0.0))
    fx = float(getattr(calibration, 'fx', f))
    fy = float(getattr(calibration, 'fy', f))
    cx = float(getattr(calibration, 'cx', 0.0))
    cy = float(getattr(calibration, 'cy', 0.0))

    if (not np.isfinite(fx)) or (not np.isfinite(fy)) or fx <= 0 or fy <= 0:
        raise RuntimeError(f"Invalid calibration focal lengths: fx={fx}, fy={fy}")

    return fx, fy, cx, cy


def estimate_camera_metric_stats(depth_channel, object_mask, calibration, depth_model):
    """
    Calculates real-world measurements for a single camera view:
    how many mm each pixel covers and the approximate image footprint in metres.
    """
    fx, fy, _, _ = get_camera_intrinsics(calibration)

    valid = object_mask & (depth_channel > 0) & np.isfinite(depth_channel)
    if not np.any(valid):
        return None

    z = depth_channel[valid].astype(np.float64)

    if depth_model == "camera_z":
        z_metric = z
    elif depth_model == "ray_distance":
        # Approximation (exact ray-distance would need per-pixel calculations).
        z_metric = z
    else:
        raise RuntimeError(f"Unsupported DEPTH_METRIC_MODEL: {depth_model}")

    gsd_x = z_metric / fx
    gsd_y = z_metric / fy

    median_depth = float(np.median(z_metric))
    height_px, width_px = depth_channel.shape
    width_m = float(width_px * (median_depth / fx))
    height_m = float(height_px * (median_depth / fy))

    return {
        'depth_model': depth_model,
        'fx_px': float(fx),
        'fy_px': float(fy),
        'median_depth_m': median_depth,
        'gsd_x_mm_per_pixel': {
            'min': float(np.min(gsd_x) * 1000.0),
            'median': float(np.median(gsd_x) * 1000.0),
            'max': float(np.max(gsd_x) * 1000.0),
        },
        'gsd_y_mm_per_pixel': {
            'min': float(np.min(gsd_y) * 1000.0),
            'median': float(np.median(gsd_y) * 1000.0),
            'max': float(np.max(gsd_y) * 1000.0),
        },
        'footprint_at_median_depth_m': {
            'width': width_m,
            'height': height_m,
        },
    }


def advanced_inpainting_with_mask(depth_map, object_mask, fill_value=FILL_VALUE, 
                                   max_iterations=MAX_INPAINTING_ITERATIONS):
    """
    Fills small gaps inside the object area without touching the background.

    Args:
        depth_map: Depth image
        object_mask: Mask showing where the object is
        fill_value: Value that marks missing pixels
        max_iterations: How many passes to run

    Returns:
        Depth map with interior gaps filled, edges left intact
    """
    filled_map = depth_map.copy().astype(np.float32)
    
    # Only identify holes WITHIN the object mask
    holes_in_object = (filled_map == fill_value) & object_mask
    
    if not np.any(holes_in_object):
        print("  No interior holes to fill within object mask")
        return depth_map
    
    print(f"  Found {np.sum(holes_in_object)} interior holes within object")
    print(f"  Background pixels: {np.sum(~object_mask)} (will not be inpainted)")
    
    # Stage 1: Distance-weighted interpolation (only within mask)
    print("    Stage 1/3: Distance-weighted interpolation")
    filled_map = distance_weighted_interpolation_masked(filled_map, object_mask, fill_value)
    
    # Stage 2: Edge-preserving smoothing (only within mask)
    print("    Stage 2/3: Edge-preserving smoothing")
    filled_map = iterative_edge_preserving_fill_masked(filled_map, holes_in_object, 
                                                        object_mask, max_iterations)
    
    # Stage 3: Morphological refinement (only within mask)
    print("    Stage 3/3: Morphological refinement")
    filled_map = morphological_refinement_masked(filled_map, holes_in_object, object_mask)
    
    if depth_map.dtype == np.uint16:
        filled_map = np.clip(filled_map, 0, 65535).astype(np.uint16)
    
    remaining_holes = np.sum((filled_map == fill_value) & object_mask)
    print(f"  ✓ Inpainting complete. Remaining interior holes: {remaining_holes}")
    
    return filled_map


def distance_weighted_interpolation_masked(depth_map, object_mask, fill_value):
    """Fill gaps by estimating values from surrounding valid pixels."""
    # Valid pixels are non-fill AND within object mask
    valid_mask = (depth_map != fill_value) & object_mask
    invalid_mask = (depth_map == fill_value) & object_mask  # Only fill holes IN object
    
    if not np.any(valid_mask) or not np.any(invalid_mask):
        return depth_map
    
    y_coords, x_coords = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    valid_coords = np.column_stack((y_coords[valid_mask], x_coords[valid_mask]))
    valid_values = depth_map[valid_mask]
    invalid_coords = np.column_stack((y_coords[invalid_mask], x_coords[invalid_mask]))
    
    try:
        # Nearest neighbor fill
        interpolated = griddata(valid_coords, valid_values, invalid_coords, method='nearest')
        depth_map[invalid_mask] = interpolated
        
        # Linear interpolation for smoothness
        interpolated_linear = griddata(valid_coords, valid_values, invalid_coords, method='linear')
        linear_valid = ~np.isnan(interpolated_linear)
        
        if np.any(linear_valid):
            invalid_y, invalid_x = invalid_coords[:, 0], invalid_coords[:, 1]
            depth_map[invalid_y[linear_valid], invalid_x[linear_valid]] = interpolated_linear[linear_valid]
    
    except Exception as e:
        print(f"    Warning: Interpolation failed: {e}")
    
    return depth_map


def iterative_edge_preserving_fill_masked(depth_map, original_holes, object_mask, max_iterations):
    """Repeatedly fills remaining gaps using nearby values, staying within the object area."""
    filled_map = depth_map.copy()
    current_holes = original_holes.copy()
    
    for iteration in range(max_iterations):
        if not np.any(current_holes):
            break
        
        remaining = np.sum(current_holes)
        print(f"      Iteration {iteration + 1}/{max_iterations}: {remaining} holes remaining")
        
        def adaptive_mean_filter(values):
            center_idx = len(values) // 2
            center_val = values[center_idx]
            
            # Only fill if this pixel is a hole AND in the object mask
            if not current_holes.flat[center_idx]:
                return center_val
            
            # Only use neighbors that are also within object mask
            valid_neighbors = values[values > 0]
            if len(valid_neighbors) == 0:
                return center_val
            
            weights = np.exp(-0.1 * np.abs(valid_neighbors - np.median(valid_neighbors)))
            weighted_mean = np.average(valid_neighbors, weights=weights)
            
            return weighted_mean
        
        filled_map = generic_filter(filled_map, adaptive_mean_filter, size=5, mode='reflect')
        
        # Update holes - only within object mask
        new_holes = original_holes & (filled_map <= 0) & object_mask
        
        if np.array_equal(current_holes, new_holes):
            print(f"      Converged after {iteration + 1} iterations")
            break
        
        current_holes = new_holes
    
    return filled_map


def morphological_refinement_masked(depth_map, original_holes, object_mask):
    """Gently smooth the filled areas so they blend with surrounding pixels."""
    from scipy.ndimage import gaussian_filter
    
    filled_regions = original_holes & object_mask
    
    if not np.any(filled_regions):
        return depth_map
    
    smoothed = gaussian_filter(depth_map.astype(np.float32), sigma=0.8)
    
    result = depth_map.copy().astype(np.float32)
    result[filled_regions] = smoothed[filled_regions]
    
    return result


def filter_outliers_for_global_range(depth_values, percentile_low=PERCENTILE_LOW, 
                                     percentile_high=PERCENTILE_HIGH):
    """Remove extreme depth values that fall outside the expected range."""
    if len(depth_values) == 0:
        return depth_values

    depth_values = depth_values[np.isfinite(depth_values)]
    if len(depth_values) == 0:
        return depth_values
    
    low_threshold = np.percentile(depth_values, percentile_low)
    high_threshold = np.percentile(depth_values, percentile_high)
    
    filtered_values = depth_values[(depth_values >= low_threshold) & 
                                   (depth_values <= high_threshold)]

    # Safety fallback if filtering removed everything (shouldn't normally happen).
    if len(filtered_values) == 0:
        filtered_values = depth_values
    
    print(f"  Outlier filtering: kept {len(filtered_values)}/{len(depth_values)} values")
    print(f"  Filtered range: {np.min(filtered_values):.3f} - {np.max(filtered_values):.3f}m")
    
    return filtered_values

# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 1: DETERMINING GLOBAL DEPTH RANGE")
print("=" * 80)

global_min_depth = float('inf')
global_max_depth = float('-inf')
all_valid_depths = []

valid_cameras = [cam for cam in chunk.cameras 
                if cam.selected and cam.enabled and cam.calibration and cam.transform]
total_cameras = len(valid_cameras)

if total_cameras == 0:
    raise RuntimeError("No valid selected/enabled cameras with calibration+transform found.")

print(f"Processing {total_cameras} cameras for global depth range...\n")

for i, camera in enumerate(valid_cameras):
    print(f"[{i+1}/{total_cameras}] Analyzing: {camera.label}")
    try:
        depth_map = chunk.model.renderDepth(camera.transform, camera.sensor.calibration)
        depth_channel = get_depth_channel(depth_map)
        depth_channel = np.nan_to_num(depth_channel, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to real-world units using the project's scale factor
        depth_channel = depth_channel * chunk_scale

        valid_depths = depth_channel[depth_channel > 0]
        
        if len(valid_depths) > 0:
            all_valid_depths.extend(valid_depths)
            print(f"  Range: {np.min(valid_depths):.3f}m - {np.max(valid_depths):.3f}m\n")
    
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        continue

if all_valid_depths:
    all_valid_depths = np.array(all_valid_depths)
    print("\nFiltering outliers from global depth range...")
    filtered_depths = filter_outliers_for_global_range(all_valid_depths)
    
    if len(filtered_depths) > 0:
        global_min_depth = np.min(filtered_depths)
        global_max_depth = np.max(filtered_depths)
else:
    raise RuntimeError("✗ No valid depth data found")

depth_range = global_max_depth - global_min_depth
if not np.isfinite(depth_range) or depth_range <= 1e-12:
    raise RuntimeError("✗ Global depth range is zero")

print(f"\n{'=' * 80}")
print(f"GLOBAL DEPTH RANGE: {global_min_depth:.3f}m to {global_max_depth:.3f}m")
print(f"Range: {depth_range:.3f}m | Precision: {depth_range/65535:.4f}m per value")
print(f"{'=' * 80}")

# Calculate real-world image dimensions from camera parameters
# Using first camera as reference (all should have same sensor)
ref_camera = valid_cameras[0]
sensor = ref_camera.sensor
image_width_px = sensor.width
image_height_px = sensor.height

# Calculate how much real-world area each pixel covers at average depth
avg_depth_m = (global_min_depth + global_max_depth) / 2

# Get pixel and focal length info from the sensor
pixel_size_mm = sensor.pixel_width if sensor.pixel_width else sensor.pixel_height
focal_length_mm = sensor.calibration.f  # retained for compatibility/logging

if (not np.isfinite(sensor.calibration.f)) or sensor.calibration.f <= 0:
    raise RuntimeError(f"Invalid camera focal length in pixels: {sensor.calibration.f}")

ref_fx, ref_fy, _, _ = get_camera_intrinsics(sensor.calibration)

# Pixel size on the object = distance / focal_length_in_pixels
gsd_x_m = avg_depth_m / ref_fx
gsd_y_m = avg_depth_m / ref_fy

# Real-world image dimensions
image_width_m = image_width_px * gsd_x_m
image_height_m = image_height_px * gsd_y_m

print(f"\n✓ Image dimensions calculated:")
print(f"  Pixels: {image_width_px} x {image_height_px}")
print(f"  GSD X/Y: {gsd_x_m * 1000:.4f} / {gsd_y_m * 1000:.4f} mm/pixel")
print(f"  Real size: {image_width_m * 1000:.2f} x {image_height_m * 1000:.2f} mm")

# Save scaling metadata
scale_metadata = {
    'global_min_depth_m': float(global_min_depth),
    'global_max_depth_m': float(global_max_depth),
    'depth_range_m': float(depth_range),
    'depth_range_mm': float(depth_range * 1000),
    'precision_m_per_value': float(depth_range/MAX_VALID_DEPTH_VALUE),
    'total_cameras_processed': total_cameras,
    'edge_protection_pixels': EDGE_PROTECTION_PIXELS,
    'inpainting_mode': 'disabled' if SKIP_INPAINTING else 'masked_interior_only',
    'chunk_transform_scale': float(chunk_scale),
    'chunk_scale_corrected': True,
    'depth_metric_model': DEPTH_METRIC_MODEL,
    'image_dimensions': {
        'width_px': int(image_width_px),
        'height_px': int(image_height_px),
        'gsd_x_mm_per_pixel': float(gsd_x_m * 1000),
        'gsd_y_mm_per_pixel': float(gsd_y_m * 1000),
        'width_mm': float(image_width_m * 1000),
        'height_mm': float(image_height_m * 1000),
        'width_m': float(image_width_m),
        'height_m': float(image_height_m),
    },
    'blender_settings': {
        'plane_width_m': float(image_width_m),
        'plane_height_m': float(image_height_m),
        'displacement_strength_m': float(depth_range),
        'color_space': 'Non-Color',
        'note': 'Depth is inverted: lower values = further from camera'
    },
    'note': 'Depth values converted to world coords by multiplying by chunk_transform_scale; 65535 is reserved as NoData/background'
}

metadata_file = os.path.join(output_folder, 'depth_scaling_metadata.json')
with open(metadata_file, 'w') as f:
    json.dump(scale_metadata, f, indent=2)
print(f"✓ Metadata saved\n")

# ============================================================================
# PHASE 2: EXPORT DEPTH MAPS WITH MASKS
# ============================================================================

print("=" * 80)
print("PHASE 2: PROCESSING AND EXPORTING DEPTH MAPS + MASKS")
print("=" * 80 + "\n")

for i, camera in enumerate(valid_cameras):
    camera_label = camera.label
    print(f"[{i+1}/{total_cameras}] Processing: {camera_label}")
    
    try:
        start_time = time.time()
        
        # Render depth map
        depth_map = chunk.model.renderDepth(camera.transform, camera.sensor.calibration)
        depth_channel = get_depth_channel(depth_map)
        depth_channel = np.nan_to_num(depth_channel, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to real-world units using the project's scale factor
        depth_channel = depth_channel * chunk_scale

        # Create object mask
        object_mask = create_object_mask(depth_channel, edge_erosion=EDGE_PROTECTION_PIXELS)
        
        # Scale valid depths into 0..65534 and reserve 65535 for NoData/background.
        depth_scaled = ((depth_channel - global_min_depth) / depth_range) * MAX_VALID_DEPTH_VALUE
        depth_scaled = np.clip(depth_scaled, 0, MAX_VALID_DEPTH_VALUE).astype(np.uint16)
        print("  ✓ Scaled to 16-bit range (0..65534; 65535 reserved for NoData)")
        
        # Invert
        depth_inverted = MAX_VALID_DEPTH_VALUE - depth_scaled
        # Set background (outside object mask) to white (65535) after inversion
        depth_inverted[~object_mask] = FILL_VALUE
        print("  ✓ Inverted depth values")

        # Inpaint only within object mask (or skip if disabled for speed)
        if SKIP_INPAINTING:
            depth_filled = depth_inverted
            print("  ⚡ Skipped inpainting (SKIP_INPAINTING=True)")
        else:
            print(f"  🔄 Starting inpainting (max {MAX_INPAINTING_ITERATIONS} iterations)...")
            depth_filled = advanced_inpainting_with_mask(depth_inverted, object_mask,
                                                         fill_value=FILL_VALUE,
                                                         max_iterations=MAX_INPAINTING_ITERATIONS)
        
        # Export depth map
        output_filename = f"{camera_label}_depth.tif"
        output_file = os.path.join(output_folder, output_filename)
        imageio.imwrite(output_file, depth_filled, format='tiff')
        print(f"  ✓ Saved depth: {output_filename}")
        
        # Export object mask as 8-bit (0=background, 255=object)
        mask_filename = f"{camera_label}_mask.tif"
        mask_file = os.path.join(output_folder, mask_filename)
        mask_8bit = (object_mask * 255).astype(np.uint8)
        imageio.imwrite(mask_file, mask_8bit, format='tiff')
        print(f"  ✓ Saved mask: {mask_filename}")
        
        # Save statistics with height measurements
        valid_pixels = np.sum(object_mask)
        filled_pixels = np.sum((depth_inverted == FILL_VALUE) & object_mask)
        
        valid_depths = depth_channel[object_mask]
        camera_min_depth = float(np.min(valid_depths)) if len(valid_depths) > 0 else 0.0
        camera_max_depth = float(np.max(valid_depths)) if len(valid_depths) > 0 else 0.0
        metric_stats = estimate_camera_metric_stats(
            depth_channel,
            object_mask,
            camera.sensor.calibration,
            DEPTH_METRIC_MODEL,
        )
        
        camera_stats = {
            'camera': camera_label,
            'object_pixels': int(valid_pixels),
            'background_pixels': int(np.sum(~object_mask)),
            'filled_interior_holes': int(filled_pixels),
            'image_dimensions': [int(depth_channel.shape[1]), int(depth_channel.shape[0])],
            'actual_depth_range_meters': {
                'min_depth': camera_min_depth,
                'max_depth': camera_max_depth,
                'range': camera_max_depth - camera_min_depth
            },
            'global_depth_range_meters': {
                'min_depth': float(global_min_depth),
                'max_depth': float(global_max_depth)
            },
            'metric_estimates': metric_stats,
            'mask_info': {
                'edge_protection_pixels': EDGE_PROTECTION_PIXELS,
                'white_in_mask': 'object/valid geometry',
                'black_in_mask': 'background/no geometry'
            }
        }
        
        stats_file = os.path.join(output_folder, f"{camera_label}_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(camera_stats, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"  ✓ Total time: {elapsed:.2f}s\n")
    
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("=" * 80)
print("EXPORT COMPLETED!")
print("=" * 80)
print(f"Output: {output_folder}")
print(f"Files per camera: depth.tif, mask.tif, stats.json")
print("=" * 80)