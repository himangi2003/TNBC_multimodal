import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import cv2
import numpy as np
from PIL import Image
from tiatoolbox.wsicore.wsireader import WSIReader  
from wholeslidedata.image.wholeslideimage import WholeSlideImage
import csv

data_path = '/lab/deasylab3/Jung/tiger/'
dir_TIFF_images = data_path + "/wsirois/wsi-level-annotations/images/"
dir_PNG_masks = data_path + "/wsirois/roi-level-annotations/tissue-cells/masks/"
dir_XML = data_path+'/wsirois/wsi-level-annotations/annotations-tissue-bcss-xmls/' 
output_root = os.path.join('/lab/deasylab3/Himangi/TNBC_multimodal', "processed_tiles")
imgs_names = sorted([f for f in os.listdir(dir_TIFF_images) if f.startswith("TCGA")])
msks_names = sorted([f for f in os.listdir(dir_PNG_masks) if f.startswith("TCGA")])
xml_names = os.listdir(dir_XML)


def load_roi_entries_from_csv(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def save_roi_coordinates_csv(imgs_names, msks_names, xml_names, prefix_filter='TCGA'):
    entries = []
    for tif_name in imgs_names:
        if not tif_name.startswith(prefix_filter):
            continue

        base = tif_name[:-4]
        relevant_masks = [m for m in msks_names if base in m[:60]]
        relevant_xml = [x for x in xml_names if base in x[:-4]]

        for mask in relevant_masks:
            try:
                match = re.search(r'\[(\d+), (\d+), (\d+), (\d+)\]', mask)
                if match:
                    x1, y1, x2, y2 = map(int, match.groups())
                    xml_file = relevant_xml[0] if relevant_xml else ""
                    entries.append({
                        "slide": tif_name,
                        "mask": mask,
                        "xml": xml_file,
                        "coord_x1": x1,
                        "coord_y1": y1,
                        "coord_x2": x2,
                        "coord_y2": y2
                    })
                else:
                    print(f"Skipping: no coordinate pattern found in {mask}")
            except Exception as e:
                print(f"Skipping malformed mask filename: {mask} ({e})")

    df = pd.DataFrame(entries)
    csv_name = f"{prefix_filter}_coordinates.csv"
    df.to_csv(csv_name, index=False)
    print(f"✅ Saved ROI coordinates CSV to: {csv_name}")


#save_roi_coordinates_csv(imgs_names, msks_names, xml_names, prefix_filter='TCGA')


def load_roi_image_and_mask(wsi_path, mask_path, coords, resolution=0.5):
    x_min, y_min, x_max, y_max = coords
    wsi = WSIReader.open(wsi_path)
    img = np.array(wsi.read_bounds(bounds=(x_min, y_min, x_max, y_max), resolution=resolution))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return img, mask


def remap_mask_classes(mask):
    """
    Remap mask from original 7-label schema to 3-class schema:
    - Tumor (1, 3) → 1
    - Stroma (2, 6) → 2
    - Rest (4, 5, 7) → 0
    """
    remapped = np.zeros_like(mask, dtype=np.uint8)

    tumor_labels = [1, 3]
    stroma_labels = [2, 6]
    rest_labels = [4, 5, 7]  # previously assumed 0 — not used now

    for label in tumor_labels:
        remapped[mask == label] = 1
    for label in stroma_labels:
        remapped[mask == label] = 2
    # Rest stays 0 by default

    return remapped

def tile_image_and_mask(img, mask, tile_size=224, stride=224):
    H, W = mask.shape
    tile_list = []
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile_img = img[y:y+tile_size, x:x+tile_size]
            tile_mask = mask[y:y+tile_size, x:x+tile_size]
            if tile_img.shape[:2] == (tile_size, tile_size):
                tile_list.append((tile_img, tile_mask, x, y))
    return tile_list

def classify_tile_by_tumor(tile_mask, threshold=0.2):
    tumor_ratio = np.mean(tile_mask == 1)
    return "tumor" if tumor_ratio >= threshold else "non_roi"

def save_tile_set(tile_img, tile_mask, overlay_dir, mask_dir, image_dir, base_name, x, y):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    name = f"{base_name}_x{x}_y{y}.png"
    Image.fromarray(tile_img).save(os.path.join(image_dir, name))
    Image.fromarray(tile_mask).save(os.path.join(mask_dir, name))
    overlay = tile_img.copy()
    overlay[tile_mask == 1] = [255, 0, 0]
    overlay[tile_mask == 2] = [0, 255, 0]
    Image.fromarray(overlay).save(os.path.join(overlay_dir, name))



def tumor_stroma_pipeline(wsi_path, mask_path, coords, base_name, output_root):
    img, mask = load_roi_image_and_mask(wsi_path, mask_path, coords)
    if img is None or mask is None or img.shape[:2] != mask.shape:
        print(f"⚠️ Skipping {base_name}: image/mask size mismatch or loading error.")
        return

    mask_remapped = remap_mask_classes(mask)
    tiles = tile_image_and_mask(img, mask_remapped)

    for img_tile, mask_tile, x, y in tiles:
        category = classify_tile_by_tumor(mask_tile)
        save_tile_set(
            img_tile,
            mask_tile,
            overlay_dir=os.path.join(output_root, "overlays", category),
            mask_dir=os.path.join(output_root, "masks", category),
            img_dir=os.path.join(output_root, "images", category),
            base_name=base_name,
            x=x, y=y
        )
    print(f"✅ Finished {base_name}. Outputs are stored in: {output_root}")

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


entries = load_roi_entries_from_csv("TCGA_coordinates.csv")[:1]

for row in entries:
    slide  = row["slide"]
    mask_f = row["mask"]
    coords = [row["coord_x1"], row["coord_y1"], row["coord_x2"], row["coord_y2"]]
    base   = mask_f.replace('.png', '')

    path_wsi  = os.path.join(dir_TIFF_images, slide)
    path_mask = os.path.join(dir_PNG_masks, mask_f)
    out_dir   = os.path.join(output_root, base)

    tumor_stroma_pipeline(path_wsi, path_mask, coords, base, out_dir)