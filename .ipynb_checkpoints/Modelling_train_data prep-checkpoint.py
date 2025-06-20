# main.py (or whatever your main script is named)

import os
import glob
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import xml.etree.ElementTree as ET
from tifffile import imwrite
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.tissuemask import OtsuTissueMasker

# Import configuration
import config


def ensure_output_dirs(base_dir):
    for label in ["tumor", "non-tumor"]:
        os.makedirs(os.path.join(base_dir, label), exist_ok=True)


def parse_asap_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    polygons = []
    for annotation in root.iter("Annotation"):
        coords = []
        for coord in annotation.iter("Coordinate"):
            x = float(coord.attrib["X"])
            y = float(coord.attrib["Y"])
            coords.append((x, y))
        if len(coords) >= 3:
            polygons.append(Polygon(coords))
    return polygons


def generate_tissue_mask(wsi):
    width_lowres, height_lowres = wsi.slide_dimensions(resolution=1.25, units="power")
    lowres_img = wsi.read_rect(location=(0, 0), size=(width_lowres, height_lowres), resolution=1.25, units="power")
    input_batch = np.expand_dims(lowres_img, axis=0)
    masker = OtsuTissueMasker()
    return masker.fit_transform(input_batch)[0]


def process_wsi(image_path, annotation_path, output_dir, tile_size, overlap, resolution):
    slide_id = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing {slide_id}...")

    wsi = WSIReader.open(image_path)
    tumor_polygons = parse_asap_xml(annotation_path)
    tumor_union = unary_union(tumor_polygons)

    width, height = wsi.slide_dimensions(resolution=resolution, units="mpp")
    tissue_mask = generate_tissue_mask(wsi)

    mask_height, mask_width = tissue_mask.shape
    stride = tile_size - overlap
    tile_metadata = []

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile_rect = box(x, y, x + tile_size, y + tile_size)

            x_mask = int(x * mask_width / width)
            y_mask = int(y * mask_height / height)
            tile_size_mask = int(tile_size * mask_width / width)

            mask_patch = tissue_mask[y_mask:y_mask + tile_size_mask, x_mask:x_mask + tile_size_mask]
            if mask_patch.shape != (tile_size_mask, tile_size_mask) or np.mean(mask_patch) < 0.5:
                continue

            tile_img = wsi.read_rect(location=(x, y), resolution=resolution, size=(tile_size, tile_size))
            label = "tumor" if tumor_union.intersects(tile_rect) else "non-tumor"

            out_path = os.path.join(output_dir, label, f"{slide_id}_x{x}_y{y}.png")
            imwrite(out_path, tile_img)

            i = y // stride
            j = x // stride
            tile_metadata.append({
                "slide_id": slide_id,
                "x": x,
                "y": y,
                "i": i,
                "j": j,
                "label": label,
                "label_numeric": 1 if label == "tumor" else 0,
                "filename": f"{slide_id}_x{x}_y{y}.png",
                "tile_path": out_path
            })

    return tile_metadata, tumor_union


def batch_process_wsi_folder():
    ensure_output_dirs(config.output_dir)
    all_metadata = []
    all_tumor_polygons = []

    image_paths = sorted(glob.glob(os.path.join(config.image_folder, "*.tif")))[:config.max_images]
    for image_path in image_paths:
        slide_id = os.path.splitext(os.path.basename(image_path))[0]
        xml_path = os.path.join(config.xml_folder, f"{slide_id}.xml")
        if not os.path.exists(xml_path):
            print(f"Warning: Missing XML for {slide_id}, skipping.")
            continue

        metadata, tumor_union = process_wsi(
            image_path, xml_path, config.output_dir,
            config.tile_size, config.overlap, config.resolution
        )
        all_metadata.extend(metadata)
        all_tumor_polygons.extend(parse_asap_xml(xml_path))

    tile_df = pd.DataFrame(all_metadata)
    tile_df.to_csv(config.metadata_csv, index=False)
    print(f"\n Done! Metadata saved to {config.metadata_csv}")

    # --- Tumor bounds check ---
    tumor_tiles_df = tile_df[tile_df["label"] == "tumor"]
    if not tumor_tiles_df.empty:
        full_union = unary_union(all_tumor_polygons)
        minx, miny, maxx, maxy = full_union.bounds

        within_bounds = (
            (tumor_tiles_df["x"] >= minx) &
            (tumor_tiles_df["x"] <= maxx) &
            (tumor_tiles_df["y"] >= miny) &
            (tumor_tiles_df["y"] <= maxy)
        )
        print("All tumor tiles within annotation bounds:", within_bounds.all())
    else:
        print(" No tumor tiles to check bounds for.")


# --- Run ---
if __name__ == "__main__":
    batch_process_wsi_folder()
