import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login

login()


import os 
import glob
png_paths = sorted(glob.glob(os.path.join("clean_tiles", "*.png")))
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model = model.eval()


transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))






device ="cpu"
def extract_deep_features_as_npy(tile_paths, output_dir="tile_npy_embeddings"):
    os.makedirs(output_dir, exist_ok=True)
    tile_coords = []

    with torch.no_grad():
        for path in tqdm(tile_paths, desc="Extracting & saving tile embeddings"):
            image = Image.open(path).convert("RGB")
            tensor = transforms(image).unsqueeze(0).to(device)
            output = model(tensor)

            class_token = output[:, 0]
            patch_tokens = output[:, 5:]
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)

            # Extract filename and coordinates
            filename = os.path.basename(path).replace(".png", "")
            parts = filename.split("_")[-2:]
            x = int(parts[0])
            y = int(parts[1])
            tile_coords.append((x, y))

            # Save as individual .npy file
            save_path = os.path.join(output_dir, f"{filename}.npy")
            np.save(save_path, embedding.squeeze(0).cpu().numpy())

    print(f"Saved {len(tile_paths)} tile embeddings to '{output_dir}/'")


extract_deep_features_as_npy(png_paths,"deep_features")