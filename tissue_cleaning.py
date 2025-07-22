import os
import numpy as np
import pandas as pd
from histolab.slide import Slide
from histolab.masks import BinaryMask
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, remove_small_holes, remove_small_objects, disk
from skimage.measure import regionprops, label
from skimage.transform import resize

# Paths
data_path = '/lab/deasylab3/Jung/tiger/'
dir_TIFF_images = os.path.join(data_path, "wsirois/wsi-level-annotations/images/")
imgs_names = sorted([f for f in os.listdir(dir_TIFF_images) if f.startswith('TCGA')])

# Custom tissue mask class
class CustomTissueMask(BinaryMask):
    def _mask(self, slide):
        thumb_rgb = np.array(slide.thumbnail)
        gray = rgb2gray(thumb_rgb)
        thresh = threshold_otsu(gray)
        binary = gray < thresh
        dilated = binary_dilation(binary, selem=disk(3))
        no_holes = remove_small_holes(dilated, area_threshold=500)
        cleaned = remove_small_objects(no_holes, min_size=1000)
        resized_mask = resize(
            cleaned.astype(np.uint8),
            slide.dimensions[::-1],
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)
        return resized_mask

# Prepare result list
results = []

for img_name in imgs_names:
    wsi_path = os.path.join(dir_TIFF_images, img_name)
    print(f"Processing: {img_name}")
    try:
        slide = Slide(wsi_path, processed_path="processed")
        mask = CustomTissueMask()
        final_mask = mask(slide)

        # Resize mask to thumbnail size
        thumbnail = np.array(slide.thumbnail)
        mask_resized = resize(
            final_mask,
            thumbnail.shape[:2],
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(bool)

        # Bounding box
        labeled = label(mask_resized)
        props = regionprops(labeled)
        if props:
            largest_region = max(props, key=lambda x: x.area)
            minr, minc, maxr, maxc = largest_region.bbox
        else:
            minr = minc = maxr = maxc = 0

        # Scale bounding box to full-res
        W, H = slide.dimensions
        thumb_h, thumb_w = thumbnail.shape[:2]
        scale_x = W / thumb_w
        scale_y = H / thumb_h

        x_min = int(minc * scale_x)
        y_min = int(minr * scale_y)
        x_max = int(maxc * scale_x)
        y_max = int(maxr * scale_y)

        results.append({
            "filename": img_name,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        })

    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        results.append({
            "filename": img_name,
            "x_min": -1,
            "y_min": -1,
            "x_max": -1,
            "y_max": -1
        })

# Save to CSV
df = pd.DataFrame(results)
output_csv_path = os.path.join(data_path, "bounding_boxes.csv")
df.to_csv(output_csv_path, index=False)
print(f"Saved bounding boxes to: {output_csv_path}")
