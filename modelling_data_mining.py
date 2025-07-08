import os
import cv2
from wholeslidedata.image.wholeslideimage import WholeSlideImage
import numpy as np

####################################################
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def extract_tcga_data(data_path):
    dir_TIFF_images_WSIROIS = data_path+'/wsirois/wsi-level-annotations/images/'
    dir_PNG_masks = data_path+'/wsirois/roi-level-annotations/tissue-bcss/masks/'    
    dir_PNG_images = data_path+'/wsirois/roi-level-annotations/tissue-bcss/images/'
    dir_XML = data_path+'/wsirois/wsi-level-annotations/annotations-tissue-bcss-xmls/'    
   
    imgs_names = os.listdir(dir_TIFF_images_WSIROIS)
    imgs_names.sort()
    msks_names = os.listdir(dir_PNG_masks)
    msks_names.sort()
    imgs_names = [i for i in imgs_names if i.startswith('TCGA')]    
    png_names = os.listdir(dir_PNG_images)
    png_names.sort()
    png_names = png_names[:-1]    
    
    imgs_msks_names_masks_coorindates = []
    for i in range(len(imgs_names)):
        temp = imgs_names[i][:-4]
        msk_names = [j for j in msks_names if temp in j[:60]]
        msk_coordinates = [j[62:-5] for j in msk_names]
        msk_coordinates = [j.split(', ') for j in msk_coordinates]
        msk_coordinates = [[int(k) for k in j] for j in msk_coordinates]        
        imgs_msks_names_masks_coorindates.append([imgs_names[i], msk_names, msk_coordinates])  
    del temp, i, msk_names,msk_coordinates

    imgs = []
    msks = []
    for i in range(len(imgs_msks_names_masks_coorindates)):      
        tif_name = imgs_msks_names_masks_coorindates[i][0]
        tif_img = WholeSlideImage(dir_TIFF_images_WSIROIS+tif_name)
        print(tif_name)        
        msk_name = imgs_msks_names_masks_coorindates[i][1][0]
        x_min_bound = imgs_msks_names_masks_coorindates[i][2][0][0]
        y_min_bound = imgs_msks_names_masks_coorindates[i][2][0][1]
        x_max_bound = imgs_msks_names_masks_coorindates[i][2][0][2]
        y_max_bound = imgs_msks_names_masks_coorindates[i][2][0][3]
        img = tif_img.get_patch(x_min_bound, y_min_bound,x_max_bound-x_min_bound, y_max_bound-y_min_bound, tif_img.spacings[0], center = False)
        msk = cv2.imread(dir_PNG_masks+msk_name)[:,:,0]
        angle,x_min,x_max,y_min,y_max = find_tcga_rotation(dir_PNG_images, dir_XML, msk_name)
        if angle !=0:
            img_rotated = rotate_image(img, -angle) 
            img_rotated = img_rotated[y_min:y_max,x_min:x_max]
            msk_rotated = rotate_image(msk, -angle) 
            msk_rotated = msk_rotated[y_min:y_max,x_min:x_max]
        else:
            img_rotated = img
            msk_rotated = msk
        imgs.append(img_rotated)
        msks.append(msk_rotated) 
        
    return imgs, msks

imgs_1, msks_1 = extract_tcga_data(data_path)
img_11 = img_1[0]
msk_11 = msks_1[0]

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


from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
import numpy as np
import cv2
import math

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def find_tcga_rotation(dir_imgs, dir_XML, name):
    PNG_img = cv2.imread(dir_imgs+name)
    shapes = PNG_img.shape
    
    xml_info = WholeSlideAnnotation(dir_XML+name[:60]+'.xml')
    
    annotations = xml_info.sampling_annotations
    
    labels_values = xml_info.labels.values
    labels_names = xml_info.labels.names
    annotations = xml_info.sampling_annotations
    
    rois = []
    roi_label = labels_names.index('roi')
    roi_label = labels_values[roi_label]
    for i in annotations:        
        if i.label.value == roi_label:
            temp = i.wkt[10:-2].split(', ')        
            temp = [i.split(' ') for i in temp]
            temp = np.array([[int(float(j)) for j in i] for i in temp])
            rois.append(temp)
    
    roi = rois[0]
    shapes = PNG_img.shape
    move_x = (max(roi[:,0])+min(roi[:,0]))/2-shapes[1]/2
    move_y = (max(roi[:,1])+min(roi[:,1]))/2-shapes[0]/2
    roi[:,0] = roi[:,0]-move_x
    roi[:,1] = roi[:,1]-move_y

    for j in range(roi.shape[0]):
        if roi[j,0]==0 and roi[j,1]!=0:
            x1,y1 = roi[j,:]
        if roi[j,0]!=0 and roi[j,1]==0:
            x2,y2 = roi[j,:]    
    angle = np.degrees(np.arctan((y2-y1)/(x2-x1)))     
    angle_image = np.degrees(np.arctan((shapes[0]-0)/(shapes[1]-0)))  

    if abs(np.round(angle_image)) != abs(np.round(angle)):        
        if abs(angle)<45:
            angle = abs(angle)
        else:
            angle = -(90-abs(angle))  
        origin = [shapes[1]/2,shapes[0]/2]            
        rois_rotated = np.zeros((5,2))
        rois_rotated[0,:] = rotate_point(origin,roi[0,:],np.radians(angle))
        rois_rotated[1,:] = rotate_point(origin,roi[1,:],np.radians(angle))
        rois_rotated[2,:] = rotate_point(origin,roi[2,:],np.radians(angle))
        rois_rotated[3,:] = rotate_point(origin,roi[3,:],np.radians(angle))
        rois_rotated[4,:] = rois_rotated[0,:]
        
        x_min = int(min(rois_rotated[:,0]))
        x_max = int(max(rois_rotated[:,0]))
        y_min = int(min(rois_rotated[:,1]))
        y_max = int(max(rois_rotated[:,1]))  

        if x_min<0:
            x_min = 0
        if x_max>shapes[1]:
            x_max = shapes[1]
        if y_min<0:
            y_min = 0
        if y_max>shapes[0]:
            y_max = shapes[0]    

    else:
        angle = 0
        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0      
    return angle,x_min,x_max,y_min,y_max


    imgs_1, msks_1 = extract_tcga_data(data_path)

data_path = '/lab/deasylab3/Jung/tiger/'
dir_TIFF_images = data_path + "/wsirois/wsi-level-annotations/images/"
dir_PNG_masks = data_path + "/wsirois/roi-level-annotations/tissue-cells/masks/"
dir_XML = data_path+'/wsirois/wsi-level-annotations/annotations-tissue-bcss-xmls/' 
output_root = os.path.join('/lab/deasylab3/Himangi/TNBC_multimodal', "processed_tiles")
imgs_names = sorted([f for f in os.listdir(dir_TIFF_images) if f.startswith("TCGA")])
msks_names = sorted([f for f in os.listdir(dir_PNG_masks) if f.startswith("TCGA")])
xml_names = os.listdir(dir_XML)

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

for img_tile, mask_tile, x, y in tiles:
    category = classify_tile_by_tumor(mask_tile)
    save_tile_set(
        img_tile,
        mask_tile,
        overlay_dir=os.path.join(output_root, "overlays", category),
        mask_dir=os.path.join(output_root, "masks", category),
        image_dir=os.path.join(output_root, "images", category),
        base_name=imgs_names[0][:-4],
        x=x, y=y
    )

################mask visualization######################################
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Point at your tile folders
output_root ='/lab/deasylab3/Himangi/TNBC_multimodal/processed_tiles'
images_dir  = os.path.join(output_root, "images",  "tumor")
masks_dir   = os.path.join(output_root, "masks",   "tumor")

# 1) Only grab actual .png files
all_files   = os.listdir(images_dir)
tile_names  = sorted([
    f for f in all_files
    if f.lower().endswith(".png") and 
       os.path.isfile(os.path.join(images_dir, f))
])[:14]

print("Will load these tiles:", tile_names)

# 2) Load images and masks
imgs = []
for fn in tile_names:
    path = os.path.join(images_dir, fn)
    try:
        imgs.append(np.array(Image.open(path)))
    except Exception as e:
        print(f"Error opening image {fn}: {e}")

msks = []
for fn in tile_names:
    path = os.path.join(masks_dir, fn)
    try:
        msks.append(np.array(Image.open(path)))
    except Exception as e:
        print(f"Error opening mask {fn}: {e}")

# 3) Define your colormap
class_labels = {
    0: 'gray',  # background or ignore
    1: 'blue',  # invasive tumor
    2: 'cyan',  # tumor-associated stroma
}
cmap   = mcolors.ListedColormap([class_labels[k] for k in sorted(class_labels)])
bounds = np.arange(len(class_labels) + 1) - 0.5
norm   = mcolors.BoundaryNorm(bounds, cmap.N)

# 4) Plot
for i in range(len(imgs)):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(imgs[i])
    plt.title("ROI Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    im = plt.imshow(msks[i], cmap=cmap, norm=norm)
    plt.title("Categorical Mask")
    plt.axis("off")
    cbar = plt.colorbar(im, ticks=sorted(class_labels.keys()))
    cbar.ax.set_yticklabels([f"Class {k}" for k in sorted(class_labels)])
    plt.tight_layout()
    plt.show()