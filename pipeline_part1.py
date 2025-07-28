from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from tiatoolbox.tools.tissuemask import OtsuTissueMasker
import numpy as np
import cv2
from tiatoolbox.wsicore.wsireader import WSIReader
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests

from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.utils import imwrite
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor


import os
data_path = '/lab/deasylab3/Jung/tiger/'
dir_TIFF_images = data_path + "/wsirois/wsi-level-annotations/images/"


imgs_names = os.listdir(dir_TIFF_images)
imgs_names.sort()
imgs_names = [i for i in imgs_names if i.startswith('TCGA')]  
wsi_path = dir_TIFF_images + imgs_names[0]
wsi_path

def crop_bounding_box_from_wsi(wsi_path, bounds, lowres_level=None):
    reader = WSIReader.open(wsi_path)
    if lowres_level is None:
        lowres_level = reader.info.level_count - 1

    downsample = reader.info.level_downsamples[lowres_level]
    crop_region = reader.read_bounds(bounds, resolution=lowres_level, units="level")
    
    return crop_region, reader, downsample


def generate_tissue_mask_otsu_tiatoolbox(thumbnail, bounds, full_dims, downsample):
    masker = OtsuTissueMasker()
    masker.fit([thumbnail])
    masks = masker.transform([thumbnail])
    lowres_mask = masks[0].astype("uint8") * 255

    h = bounds[3] - bounds[1]
    w = bounds[2] - bounds[0]

    fullres_mask = cv2.resize(lowres_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros(full_dims[::-1], dtype="uint8")
    canvas[bounds[1]:bounds[3], bounds[0]:bounds[2]] = fullres_mask

    return canvas


def tile_and_save_patches(reader, mask_array, out_dir, slide_id,
                          patch_size=224, stride=112, min_mask_ratio=0.5):
    os.makedirs(out_dir, exist_ok=True)
    mask_reader = VirtualWSIReader(np.expand_dims(mask_array, axis=2))

    extractor = SlidingWindowPatchExtractor(
        input_img=reader,
        patch_size=patch_size,
        stride=stride,
        resolution=0,
        units="level",
        input_mask=mask_reader,
        within_bound=True,
        min_mask_ratio=min_mask_ratio,
    )

    coords = extractor.locations_df[["x", "y"]]
    metadata = []

    for i, patch in enumerate(tqdm(extractor, desc=f"Tiling {slide_id}")):
        x, y = int(coords.iloc[i]["x"]), int(coords.iloc[i]["y"])
        filename = f"{slide_id}_tile_{i}_x{x}_y{y}.png"
        Image.fromarray(patch).save(os.path.join(out_dir, filename))
        metadata.append({"index": i, "filename": filename, "x": x, "y": y, "slide": slide_id})

    return metadata