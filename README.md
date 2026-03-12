# High-Resolution Depth Map Exporter for Agisoft Metashape

A Python script that exports 16-bit depth maps from photogrammetric meshes in Agisoft Metashape. Designed for heritage imaging workflows where calibrated depth data is needed alongside standard photographic outputs.

## What it does

For each camera position in your Metashape project, the script exports:

- A **16-bit depth map** (TIFF) with metric-accurate values
- An **object mask** showing where valid surface data exists
- A **statistics file** (JSON) with per-camera measurements
- A **global metadata file** (JSON) with project-wide depth range and scaling info

The depth maps use a shared global scale across all cameras, so values are directly comparable between views. Background areas are clearly marked (value 65535), and interior holes in the surface can optionally be filled in.

## Why it exists

Metashape does not natively export per-camera depth maps from meshes in this format. This script was developed to extract calibrated depth data for:

- Merging photogrammetry depth with RTI/photometric stereo detail
- Producing tactile reproductions of heritage objects (e.g. via CNC milling)
- Any workflow that needs consistent, metric depth images aligned to the original photographs

## Requirements

This script runs inside the Agisoft Metashape Python environment. The following additional libraries must be available in that environment:

- numpy
- imageio
- scipy

See `requirements.txt` for versions.

## How to use it

1. Open your Metashape project and make sure the target chunk is active.
2. Select and enable the cameras you want to export.
3. Run the script from the Metashape Python console or via Tools > Run Script.
4. Choose an output folder when prompted.

The script will first scan all selected cameras to determine the global depth range, then export each camera's depth map, mask, and statistics.

**Note:** This script is slow — Metashape will appear as "Not Responding" for extended periods during export. This is normal, just let it run. If your mesh has holes, the inpainting step will slow things down considerably; you can set `SKIP_INPAINTING = True` at the top of the script for faster exports at the cost of unfilled gaps.

## Configuration

At the top of the script you can adjust:

- `SKIP_INPAINTING` — set to `True` for fast preview exports (skips hole filling)
- `MAX_INPAINTING_ITERATIONS` — controls quality of interior hole filling
- `EDGE_PROTECTION_PIXELS` — border margin to prevent edge artefacts during inpainting

## Output files

| File | Description |
|------|-------------|
| `<camera>_depth.tif` | 16-bit depth map (0-65534 = depth, 65535 = background) |
| `<camera>_mask.tif` | 8-bit object mask (255 = object, 0 = background) |
| `<camera>_stats.json` | Per-camera depth range and pixel measurements |
| `depth_scaling_metadata.json` | Global depth range, scaling, and image dimensions |

## Related tools

This script is the first step in a multi-source depth fusion pipeline:

1. **Export depth maps** from Metashape (this script)
2. **Integrate RTI/photometric stereo normal maps into depth maps** using a modified version of BINI adapted for large images (separate project, coming soon)
3. **Merge PG + PS depth maps** using [RTI-photogrammetry-depthmaps-fusion](https://github.com/XaviAure/RTI-photogrammetry-depthmaps-fusion)

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).
