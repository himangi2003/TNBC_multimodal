from matplotlib.colors import ListedColormap
from PIL import Image
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from tiatoolbox.wsicore.wsireader import WSIReader

import logging
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
warnings.filterwarnings("ignore")


from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.utils.misc import imread, imwrite
import matplotlib.pyplot as plt
import os
import csv
roi_img_dir = '/lab/deasylab3/Himangi/tnbc_data/roi_imagesroi_TCGA-A1-A0SP-01Z-00-DX1.20D689C6-EFA5-4694-BE76-24475A89ACC0_3000_24862_9000_30896.png'
reader = WSIReader.open(roi_img_dir)
print(reader)  # no
# 1. Load image from path
input_img = imread(roi_img_dir) 


import glob
img_paths = sorted(glob.glob(roi_img_dir))



from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,)

# Define your segmentor
bcc_segmentor = SemanticSegmentor(
    pretrained_model="fcn_resnet50_unet-bcss",
    num_loader_workers=4,
    batch_size=4,
    auto_generate_mask=False,
)



bcc_roi_ioconfig = IOSegmentorConfig(
    input_resolutions=[{"units": "mpp", "resolution": 0.25}],   # or a value you assume
    output_resolutions=[{"units": "mpp", "resolution": 0.25}],
    patch_input_shape=[512, 512],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    save_resolution={"units": "mpp", "resolution": 0.25}
)


wsi_output = bcc_segmentor.predict(
    imgs=img_paths,
    masks=None,
    save_dir="sample_wsi_results/",
    mode="tile",  # <- Use tile mode for ROI images
    ioconfig=bcc_roi_ioconfig,
    device="cpu",
    crash_on_exception=True,
)


#tumor plot 
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import measure
from skimage.draw import polygon_perimeter
import os

# Assume wsi_output[0] = (image_name, prediction_path_prefix)
image_name, prediction_path_prefix = wsi_output[0]

# Load raw prediction (.npy) from disk
tile_prediction_raw = np.load(prediction_path_prefix + ".raw.0.npy")
print("Raw prediction shape:", tile_prediction_raw.shape)

# Convert softmax to class labels (argmax)
tile_prediction = np.argmax(tile_prediction_raw, axis=-1)
print("Processed prediction shape:", tile_prediction.shape)

# Load the original ROI image
img_path = os.path.join("path_to_your_images", image_name)  # change as needed
tile = imread(img_path)
if tile.ndim == 2:
    tile = np.stack([tile]*3, axis=-1)
elif tile.shape[2] == 4:
    tile = tile[:, :, :3]  # Remove alpha channel if present

# Get tumor mask (class 0)
tumor_mask = (tile_prediction == 0).astype(np.uint8)

# Find contours
contours = measure.find_contours(tumor_mask, level=0.5)

# Create overlay with red contours
tile_overlay = tile.copy()
for contour in contours:
    contour = np.round(contour).astype(np.int32)
    rr, cc = polygon_perimeter(contour[:, 0], contour[:, 1], shape=tile_overlay.shape)
    tile_overlay[rr, cc] = [255, 0, 0]  # Red

# Optional: blend tumor heatmap
alpha = 0.4
tumor_overlay = tile.copy()
tumor_overlay[tumor_mask == 1] = (
    (1 - alpha) * tile[tumor_mask == 1] + alpha * np.array([255, 0, 0])
).astype(np.uint8)



plt.figure(figsize=(8, 4))  # Wider for side-by-side

plt.subplot(1, 2, 1)
plt.imshow(tile)
plt.title("Original ROI")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(tumor_overlay)
plt.title("Tumor Overlay (Heatmap)")
plt.axis("off")

plt.tight_layout()
plt.show()


# contouring 

import cv2

# Make sure image is RGB and uint8
tile_overlay = tile.copy()
if tile_overlay.ndim == 2:
    tile_overlay = np.stack([tile_overlay] * 3, axis=-1)
elif tile_overlay.shape[2] == 4:
    tile_overlay = tile_overlay[:, :, :3]

tile_overlay = tile_overlay.copy()

# Convert contours to OpenCV format and draw them
for contour in contours:
    contour = np.round(contour).astype(np.int32)
    if len(contour) < 3:
        continue  # Need at least 3 points for a polygon

    # OpenCV expects shape: (n_points, 1, 2)
    contour_cv = contour[:, [1, 0]].reshape(-1, 1, 2)  # Flip (row, col) -> (x, y)

    # Draw contour with thickness 2 and red color
    cv2.polylines(tile_overlay, [contour_cv], isClosed=True, color=(255, 0, 0), thickness=20)

# Show overlay
plt.figure(figsize=(12, 8))
plt.imshow(tile_overlay)
plt.title("Tumor Contour Overlay (cv2.polylines)")
plt.axis("off")
plt.show()


## saving contouring
import json
from skimage import measure

# Label tumor mask for region-based analysis
from skimage.measure import label, regionprops

labeled_mask = label(tumor_mask)

regions = regionprops(labeled_mask)

# Export basic region info to JSON
features = []
for i, region in enumerate(regions):
    if region.area < 100:  # filter out noise
        continue
    features.append({
        "id": i,
        "area": region.area,
        "perimeter": region.perimeter,
        "centroid": region.centroid,
        "bbox": region.bbox,
        "eccentricity": region.eccentricity,
        "solidity": region.solidity,
        "extent": region.extent,
        "compactness": 4 * np.pi * region.area / (region.perimeter ** 2 + 1e-6),  # avoid div by 0
    })

with open("tumor_regions_morphology.json", "w") as f:
    json.dump(features, f, indent=2)

print(f"✅ Exported {len(features)} tumor regions with morphological features.")


### plotting 
import matplotlib.pyplot as plt

areas = [f["area"] for f in features]
eccentricities = [f["eccentricity"] for f in features]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(areas, bins=50)
plt.title("Tumor Region Areas")

plt.subplot(1, 2, 2)
plt.hist(eccentricities, bins=50)
plt.title("Tumor Region Eccentricity")

plt.tight_layout()
plt.show()
