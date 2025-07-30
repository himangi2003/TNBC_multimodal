
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor

data_path = '/lab/deasylab3/Jung/tiger/'
dir_TIFF_images = data_path + "/wsirois/wsi-level-annotations/images/"


imgs_names = os.listdir(dir_TIFF_images)
imgs_names.sort()
imgs_names = [i for i in imgs_names if i.startswith('TCGA')]  
wsi_path = dir_TIFF_images + imgs_names[0]
wsi_path



def extract_and_save_patches(
    wsi_path,
    x_min,
    y_min,
    x_max,
    y_max,
    out_dir="tiles",
    patch_size=(224, 224),
    stride=(224, 224),
    min_mask_ratio=0.5,
):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pipeline_tiles.csv")

    # Step 1: Open WSI
    wsi = WSIReader.open(wsi_path)
    wsi_dims = wsi.slide_dimensions(resolution=0, units="level")
    bounds = (x_min, y_min, x_max, y_max)
    
    # Generate tissue mask only in bounding box
    tissue_mask = generate_tissue_mask(wsi, bounds)
    
    # Combine into full mask
    combined_mask = combine_bbox_and_tissue_mask(wsi_dims, x_min, y_min, tissue_mask)



    # Step 2: Create bounding box binary mask
    input_mask = (combined_mask > 0).astype(np.uint8)
    input_mask = np.expand_dims(input_mask, axis=-1)  # (H, W, 1)



    # Step 4: Initialize patch extractor
    extractor = SlidingWindowPatchExtractor(
        input_img=wsi,
        patch_size=(224, 224),
        stride=(224, 224),
        resolution=0,
        units="level",
        input_mask=input_mask,        # Direct NumPy array
        within_bound=True,
        min_mask_ratio=0.5,
    )


    # Step 5: Extract and save patches
    coords = extractor.locations_df[["x", "y"]]
    metadata = []

    for i, patch in enumerate(tqdm(extractor, desc="Extracting patches")):
        x, y = int(coords.iloc[i]["x"]), int(coords.iloc[i]["y"])
        filename = f"tile_{i}_{x}_{y}.png"
        save_path = os.path.join(out_dir, filename)
        Image.fromarray(patch).save(save_path)

        # Read matching region from the binary mask to calculate tissue ratio
        mask_patch = mask_reader.read_rect(location=(x, y),
                                           size=patch_size,
                                           resolution=0,
                                           units="level")

        tissue_ratio = np.mean(mask_patch.astype(np.float32))  # [0.0, 1.0]

        metadata.append({
            "index": i,
            "filename": filename,
            "x": x,
            "y": y,
            "tissue_ratio": tissue_ratio
        })

    # Step 6: Save metadata
    pd.DataFrame(metadata).to_csv(csv_path, index=False)

    print(f"✅ Saved {len(metadata)} patches to {out_dir}")
    print(f"📄 Patch metadata CSV saved to {csv_path}")

 x_min,y_min,x_max,y_max = 8209,200,59972,34836

 extract_and_save_patches(
    wsi_path,
    x_min,
    y_min,
    x_max,
    y_max,
    out_dir="Output5",
    patch_size=(224, 224),
    stride=(224, 224),
    min_mask_ratio=0.5,
)



 
import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, remove_small_holes, label

def is_pen_dominated(tile_rgb, sat_thresh=30, hue_range_thresh=10):
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    high_sat = hsv[:, :, 1] > sat_thresh
    if np.sum(high_sat) / high_sat.size < 0.6:
        return False
    hue_vals = hsv[:, :, 0][high_sat]
    if len(hue_vals) == 0:
        return False
    hue_range = np.percentile(hue_vals, 95) - np.percentile(hue_vals, 5)
    return hue_range < hue_range_thresh

def get_tissue_mask_basic(rgb_image, deconvolve_first=True, sigma=1.2, min_size=300):
    if deconvolve_first:
        hed = rgb2hed(rgb_image)
        hema = -hed[:, :, 0]
        hema = (hema - np.min(hema)) / (np.max(hema) - np.min(hema) + 1e-6)
        gray_image = hema
    else:
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) / 255.0
    smooth = gaussian(gray_image, sigma=sigma)
    thresh = threshold_otsu(smooth)
    mask = smooth > thresh
    mask = remove_small_objects(mask, min_size=min_size)
    mask = remove_small_holes(mask, area_threshold=min_size)
    labeled = label(mask)
    return labeled, mask.astype(np.uint8)

def is_good_tissue_tile(image_path):
    try:
        tile = cv2.imread(image_path)
        if tile is None or tile.shape[:2] != (224, 224):
            return False
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
        if np.mean(hsv[:, :, 1]) < 15:
            return False
        if is_pen_dominated(tile_rgb):
            return False
        _, tissue_mask = get_tissue_mask_basic(tile_rgb)
        return np.sum(tissue_mask) > 0.12 * tissue_mask.size
    except Exception:
        return False

def process_extracted_tiles_reindex_only(tile_dir, metadata_csv, output_dir):
    df_original = pd.read_csv(metadata_csv)
    os.makedirs(output_dir, exist_ok=True)

    df_original["tile_path"] = df_original["filename"].apply(lambda x: os.path.join(tile_dir, x))
    df_original["is_tissue"] = df_original["tile_path"].apply(is_good_tissue_tile)

    df_original.to_csv(os.path.join(tile_dir, "tile_classification_with_coords.csv"), index=False)

    filtered_df = df_original[df_original["is_tissue"]].copy().reset_index(drop=True)

    clean_metadata = []
    for idx, row in filtered_df.iterrows():
        old_path = row["tile_path"]
        match = re.match(r"tile_\d+_(\d+)_(\d+)\.png", row["filename"])
        if match:
            x, y = match.groups()
            new_filename = f"tile_{idx}_{x}_{y}.png"
        else:
            new_filename = f"tile_{idx:05d}.png"

        new_path = os.path.join(output_dir, new_filename)
        Image.open(old_path).save(new_path)

        clean_metadata.append({
            "new_index": idx,
            "new_filename": new_filename,
            "original_filename": row["filename"],
            "x": row["x"],
            "y": row["y"]
        })

    df_clean = pd.DataFrame(clean_metadata)
    df_clean.to_csv(os.path.join(output_dir, "filtered_tissue_tiles.csv"), index=False)

    print("✅ Processing complete.")
    print(f"📄 Classification saved: {os.path.join(tile_dir, 'tile_classification_with_coords.csv')}")
    print(f"📄 Filtered tile metadata: {os.path.join(output_dir, 'filtered_tissue_tiles.csv')}")
    print(f"🖼️  Clean tissue tiles saved in: {output_dir}")

# Example usage:
process_extracted_tiles_reindex_only("Output5", "Output5/pipeline_tiles.csv", "clean_tiles")
