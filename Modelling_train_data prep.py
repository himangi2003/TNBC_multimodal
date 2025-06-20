import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import imwrite
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import os
os.listdir('/')
image_path = "/lab/deasylab3/Jung/tiger/wsibulk/images/103S.tif"
annotation_path =  "/lab/deasylab3/Jung/tiger/wsibulk/annotations-tumor-bulk/xmls/103S.xml"

output_dir = "output_tiles"
os.makedirs(output_dir, exist_ok=True)

tile_size = 224
level = 0

for label in ["tumor", "non-tumor"]:
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)
from tiatoolbox.wsicore.wsireader import WSIReader
# --- Process the Slide ---

wsi = WSIReader.open(image_path)
level = 0

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


tumor_polygons = parse_asap_xml(annotation_path )
tumor_union = unary_union(tumor_polygons)

dims = wsi.slide_dimensions(resolution=resolution, units="mpp")
width, height = dims
tile_metadata = []

for y in range(0, height, tile_size):
    for x in range(0, width, tile_size):
        x_real = x * (2 ** level)
        y_real = y * (2 ** level)
        tile_rect = box(x_real, y_real, x_real + tile_size * (2 ** level), y_real + tile_size * (2 ** level))

        tile_img = wsi.read_rect((x_real, y_real), resolution=resolution, size=(tile_size, tile_size))

        # Otsu threshold
        gray_tile = rgb2gray(np.asarray(tile_img)[..., :3])
        otsu_thresh = threshold_otsu(gray_tile)
        if np.mean(gray_tile > otsu_thresh) > 0.5:
            continue

        # Label tile
        label = "non-tumor"
        if tumor_union.intersects(tile_rect):
            label = "tumor"

        out_path = os.path.join(output_dir, label, f"{slide_id}_x{x}_y{y}.png")
        imwrite(out_path, tile_img)

        tile_metadata.append({
            "slide_id": slide_id,
            "x": x,
            "y": y,
            "label": label,
            "tile_path": out_path
        })



tile_df = pd.DataFrame(tile_metadata)
tile_df.to_csv("tile_labels_metadata.csv", index=False)