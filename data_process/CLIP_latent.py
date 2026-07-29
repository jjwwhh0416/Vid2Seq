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

# jpg 정렬
images = sorted(image_dir.glob("*.jpg"))

# nuScenes는 2FPS → 1FPS
images = images[::2]

features = []

with torch.no_grad():
    for img_path in tqdm(images):

        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        feat = model.get_image_features(**inputs)

        # L2 normalize (CLIP 기본 사용법)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        features.append(feat.cpu().numpy()[0])

features = np.stack(features)

print(features.shape)

np.save("scene001.npy", features)