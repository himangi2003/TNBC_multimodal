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
from tiatoolbox.wsicore.wsireader import WSIReader
import glob
"""Import modules required to run the Jupyter notebook."""

# Clear logger to use tiatoolbox.logger
import logging
import warnings

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from tiatoolbox import logger
from tiatoolbox.models.architecture.unet import UNetModel
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)
from tiatoolbox.utils.misc import download_data, imread
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.wsicore.wsireader import WSIReader

mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
warnings.filterwarnings("ignore")

data_path = '/lab/deasylab3/Jung/tiger/'
dir_TIFF_images = data_path + "/wsirois/wsi-level-annotations/images/"
dir_PNG_masks = data_path + "/wsirois/roi-level-annotations/tissue-cells/masks/"
dir_XML = data_path+'/wsirois/wsi-level-annotations/annotations-tissue-bcss-xmls/' 


imgs_names = os.listdir(dir_TIFF_images_WSIROIS)
imgs_names.sort()
msks_names = os.listdir(dir_PNG_masks)
msks_names.sort()
imgs_names = [i for i in imgs_names if i.startswith('TCGA')]    
png_names = os.listdir(dir_PNG_images)
png_names.sort()
png_names = png_names[:-1] 
coordinates = pd.read_csv("TCGA_roi_coordinates.csv")




bounds = [3000,24862,9000,30896]
reader = WSIReader.open(dir_TIFF_images + imgs_names[1] )
img = reader.read_bounds(bounds, resolution=0, units="level")

plt.imshow(img)
plt.axis("off")
plt.show() 

from PIL import Image
import numpy as np
import os

# Ensure img is a NumPy array
if not isinstance(img, np.ndarray):
    img = np.array(img)

# Base directory for saving
save_dir = "roi_images"
os.makedirs(save_dir, exist_ok=True)

# Construct filename
base_name = os.path.splitext(imgs_names[1])[0]
x1, y1, x2, y2 = bounds
filename = f"roi_{base_name}_{x1}_{y1}_{x2}_{y2}.png"

# Full save path
save_path = os.path.join(save_dir, filename)

# Save the image
Image.fromarray(img).save(save_path)
print(f"✅ ROI saved to: {save_path}")


# semantic segmentic on ROI for tissue'




# Tile prediction
bcc_segmentor = SemanticSegmentor(
    pretrained_model="fcn_resnet50_unet-bcss",
    num_loader_workers=4,
    batch_size=4,
)

png_paths = sorted(glob.glob(os.path.join(save_dir, "*.png")))
output = bcc_segmentor.predict(
    png_paths,
    save_dir="sample_tile_results2/",
    mode="tile",
    resolution=1.0,
    units="baseline",
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    device="cuda",
    crash_on_exception=True,
)

import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from matplotlib.colors import ListedColormap
from PIL import Image
import logging

def visualize_and_save_segmentation(tile_path, prediction_output, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    basename = os.path.basename(tile_path).replace(".png", "")
    pred_path = prediction_output[1] + ".raw.0.npy"

    if not os.path.exists(pred_path):
        logger.error(f"Missing prediction file: {pred_path}")
        return

    from PIL import Image
    tile = np.array(Image.open(tile_path))

    pred_raw = np.load(pred_path)
    pred_mask = np.argmax(pred_raw, axis=-1)

    logger.info(f"Loaded: {basename}, prediction shape: {pred_raw.shape}")
    label_names_dict = {
    0: "Tumour",
    1: "Stroma",
    2: "Inflamatory",
    3: "Necrosis",
    4: "Others",}

    class_colors = [
    (255, 0, 0),      # Tumour - red
    (0, 255, 0),      # Stroma - green
    (0, 0, 255),      # Inflammatory - blue
    (255, 255, 0),    # Necrosis - yellow
    (128, 128, 128),  # Others - gray
    ]


    # ── 1. Save raw class maps ─────────────────────────────
    fig = plt.figure(figsize=(12, 3))
    for i in range(pred_raw.shape[-1]):
        ax = plt.subplot(1, pred_raw.shape[-1], i + 1)
        plt.imshow(pred_raw[:, :, i], cmap="viridis")
        plt.title(label_names_dict.get(i, f"Class {i}"))
        ax.axis("off")
    fig.suptitle(f"Raw Predictions – {basename}")
    fig.savefig(os.path.join(save_dir, f"{basename}_classmaps.png"), bbox_inches="tight")
    plt.close(fig)

    # ── 2. Save processed class mask ───────────────────────
    cmap = ListedColormap(np.array(class_colors) / 255.0)
    fig2 = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(tile)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap=cmap)
    plt.title("Segmentation")
    plt.axis("off")
    fig2.suptitle(f"Prediction – {basename}")
    fig2.savefig(os.path.join(save_dir, f"{basename}_segmentation.png"), bbox_inches="tight")
    plt.close(fig2)

    # ── 3. Save colored overlay ────────────────────────────
    overlay = tile.copy()
    for cls_id, color in enumerate(class_colors):
        mask = pred_mask == cls_id
        overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array(color)).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(save_dir, f"{basename}_overlay.png"))

    # ── 4. Save raw and mask as .npy (optional) ────────────
    np.save(os.path.join(save_dir, f"{basename}_mask.npy"), pred_mask)
    np.save(os.path.join(save_dir, f"{basename}_raw.npy"), pred_raw)

    logger.info(f"✅ Saved prediction visualizations for {basename}")




visualize_and_save_segmentation(
    tile_path=png_paths[0],
    prediction_output=output[0],
    save_dir="/lab/deasylab3/Himangi/TNBC_multimodal/pred"
)

"""Import modules required to run the Jupyter notebook."""

# Clear logger to use tiatoolbox.logger
import logging
import warnings

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tiatoolbox import logger
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import download_data, imread

# We need this function to visualize the nuclear predictions
from tiatoolbox.utils.visualization import (
    overlay_prediction_contours,
)
from tiatoolbox.wsicore.wsireader import WSIReader

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
plt.rcParams.update({"font.size": 5})


# cell segmentation 
inst_segmentor = NucleusInstanceSegmentor(
    pretrained_model="hovernet_fast-pannuke",
    num_loader_workers=2,
    num_postproc_workers=2,
    batch_size=4,
)


tile_output = inst_segmentor.predict(
    png_paths,
    save_dir="sample_tile_results3/",
    mode="tile",
    device="cuda",
    crash_on_exception=False,
)

# instance cell segmentation prediction 
tile_preds = joblib.load(f"{tile_output[0][1]}.dat")
logger.info(f"Number of detected nuclei: {len(tile_preds)}")

# Extracting the nucleus IDs and select the first one
nuc_id_list = list(tile_preds.keys())
selected_nuc_id = nuc_id_list[0]
logger.info(f"Nucleus prediction structure for nucleus ID: {selected_nuc_id}")
sample_nuc = tile_preds[selected_nuc_id]
sample_nuc_keys = list(sample_nuc)
logger.info(
    "Keys in the output dictionary: [%s, %s, %s, %s, %s]",
    sample_nuc_keys[0],
    sample_nuc_keys[1],
    sample_nuc_keys[2],
    sample_nuc_keys[3],
    sample_nuc_keys[4],
)
logger.info(
    "Bounding box: (%d, %d, %d, %d)",
    sample_nuc["box"][0],
    sample_nuc["box"][1],
    sample_nuc["box"][2],
    sample_nuc["box"][3],
)
logger.info(
    "Centroid: (%d, %d)",
    sample_nuc["centroid"][0],
    sample_nuc["centroid"][1],
)

# visualization of cell segmentation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# Reading the original image
tile_img = imread(save_path)

# your original dict
color_dict = {
    0: ("background", (255, 165, 0)),
    1: ("neoplastic epithelial", (255, 0, 0)),
    2: ("Inflammatory", (255, 255, 0)),
    3: ("Connective", (0, 255, 0)),
    4: ("Dead", (0, 0, 0)),
    5: ("non-neoplastic epithelial", (0, 0, 255)),
}

# generate overlay
overlaid = overlay_prediction_contours(
    canvas=tile_img,
    inst_dict=tile_preds,
    draw_dot=False,
    type_colours=color_dict,
    line_thickness=2,
)

# display with legend
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(tile_img)
ax1.axis("off")
ax1.set_title("Original")

ax2.imshow(overlaid)
ax2.axis("off")
ax2.set_title("Segmentation Overlay")

# build a legend from your color_dict
import matplotlib.patches as mpatches
handles = []
for _, (label, rgb) in color_dict.items():
    # normalize RGB to [0,1]
    handles.append(
        mpatches.Patch(color=np.array(rgb) / 255.0, label=label)
    )
ax2.legend(
    handles=handles,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.
)

plt.tight_layout()
plt.show()
