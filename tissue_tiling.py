
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