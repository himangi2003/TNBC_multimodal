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




data_path = '/lab/deasylab3/Jung/tiger/'
dir_TIFF_images = data_path + "/wsirois/wsi-level-annotations/images/"
dir_PNG_masks = data_path + "/wsirois/roi-level-annotations/tissue-cells/masks/"

imgs_names = sorted([f for f in os.listdir(dir_TIFF_images) if f.startswith("TCGA")])[0]
msks_names = sorted([f for f in os.listdir(dir_PNG_masks) if f.startswith("TCGA")])[0]

roi_data = []

for tif_file in imgs_names:
    base = tif_file[:-4]
    related_masks = [m for m in msks_names if base in m[:25]]
    coords = [[int(k) for k in m[27:-5].split(', ')] for m in related_masks]
    roi_data.append([tif_file, related_masks, coords])



imgs, msks = [], []

for tif_file, msk_list, coords in roi_data:
    wsi = WholeSlideImage(dir_TIFF_images + tif_file)
    for msk_file, (x1, y1, x2, y2) in zip(msk_list, coords):
        img_roi = wsi.get_patch(x1, y1, x2 - x1, y2 - y1, wsi.spacings[0], center=False)
        mask = cv2.imread(dir_PNG_masks + msk_file)[:, :, 0]
        imgs.append(img_roi)
        msks.append(mask)


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

msks_tc = [remap_mask_classes(i) for i in msks_tc]

# Define class labels and their corresponding colors
class_labels = {
    0: 'gray',     # background or ignore
    1: 'blue',       # invasive tumor
    2: 'cyan',     # tumor-associated stroma
}

# Create colormap and normalization
cmap = mcolors.ListedColormap([class_labels[k] for k in sorted(class_labels)])
bounds = np.arange(len(class_labels)+1) - 0.5
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot loop
for i in range(14):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(imgs[i])
    plt.title("ROI Image")

    plt.subplot(1, 2, 2)
    im = plt.imshow(msks[i], cmap=cmap, norm=norm)
    plt.title("Categorical Mask")
    cbar = plt.colorbar(im, ticks=sorted(class_labels.keys()))
    cbar.ax.set_yticklabels([f"Class {k}" for k in sorted(class_labels)])
    plt.show()


##################################semantic segmentation####################################################

logger = logging.getLogger(__name__)
label_names_dict = {
    0: "Tumour",
    1: "Stroma",
    2: "Inflammatory",
    3: "Necrosis",
    4: "Others",
}
class_colors = [
    (255, 0, 0),      # Tumour - red
    (0, 255, 0),      # Stroma - green
    (0, 0, 255),      # Inflammatory - blue
    (255, 255, 0),    # Necrosis - yellow
    (128, 128, 128),  # Others - gray
]


import os
import glob

images_dir = "/lab/deasylab3/Himangi/TNBC_multimodal/processed_tiles/images/tumor"
tile_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))

output_dir = os.path.join(output_root, "segmentor_results_f")


# Initialize segmentor once
bcc_segmentor = SemanticSegmentor(
    pretrained_model="fcn_resnet50_unet-bcss",
    num_loader_workers=4,
    batch_size=4,
)

# ── 3) Run prediction on all tumor tiles at once ───────────────────────────────
output = bcc_segmentor.predict(
    tile_paths,                          # All PNG tiles at once
    save_dir=output_dir,                            # Where to save outputs
    mode="tile",                                    # Use tile-wise inference
    resolution=0.5,                                 # Pixel spacing; irrelevant here if input is PNG
    units="mpp",                               # Keep 'baseline' for raw pixel spacing
    patch_input_shape=[TILE_SIZE, TILE_SIZE],       # 224x224 input
    patch_output_shape=[TILE_SIZE, TILE_SIZE],      # 224x224 output
    stride_shape=[TILE_SIZE, TILE_SIZE],            # No overlap
    device=device,
    crash_on_exception=False,
)

print("✅ Done – segmentation results saved to:", output_dir)




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
    tile_path=tile_paths[0],
    prediction_output=output[0],
    save_dir="/lab/deasylab3/Himangi/TNBC_multimodal/pred_viz"
)

#################################