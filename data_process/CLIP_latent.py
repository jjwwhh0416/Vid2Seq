import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14"
).to(device)
model.eval()

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14"
)

image_dir = Path("/mnt/ddn/prod-shared/datasets/DriveLM/nuscenes/samples/CAM_FRONT")

# 사용할 scene
scene_prefix = "n015-2018-07-24-10-42-41+0800"

# 해당 scene의 이미지만 선택
images = sorted(image_dir.glob(f"{scene_prefix}*.jpg"))

print(f"Total images: {len(images)}")

# nuScenes는 2 FPS → Vid2Seq는 1 FPS
images = images[::2]

print(f"Using images: {len(images)}")

features = []

with torch.no_grad():
    for img_path in tqdm(images):
        image = Image.open(img_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        feat = model.get_image_features(**inputs)

        # CLIP feature normalization
        feat = feat / feat.norm(dim=-1, keepdim=True)

        features.append(feat.cpu().numpy()[0])

features = np.stack(features)

print("Feature shape:", features.shape)

np.save("scene001.npy", features)