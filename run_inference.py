from config import DATA_DIR, RESULTS_DIR, MODEL_PATH

import os
import json
import random
import time


import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =========================================================
# MAIN SETTINGS
# =========================================================


IMAGE_SIZE = 256
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_WORKERS = 0

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================
# REPODUCIBILITY
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# DATASET
# =========================================================
def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)

def collect_image_paths(root_dir: str):
    paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if is_image_file(file):
                paths.append(os.path.join(root, file))
    paths.sort()
    return paths

class MarsAcceptableDataset(Dataset):
    def __init__(self, image_paths, image_size=256):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                x = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Error loading image: {path} | {e}")

        return x, path

# =========================================================
# MODEL
# =========================================================
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# =========================================================
# UTILS
# =========================================================
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

def compute_batch_reconstruction_errors(inputs, outputs):
    return torch.mean(torch.abs(inputs - outputs), dim=(1, 2, 3))

# =========================================================
# DATA
# =========================================================
all_image_paths = collect_image_paths(DATA_DIR)
print(f"Collected images: {len(all_image_paths)}")

if len(all_image_paths) == 0:
    raise ValueError("No images found in DATA_DIR.")

dataset = MarsAcceptableDataset(
    image_paths=all_image_paths,
    image_size=IMAGE_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# =========================================================
# LOAD MODEL
# =========================================================
model = ConvAutoencoder().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded model from: {MODEL_PATH}")

# =========================================================
# INFERENCE
# =========================================================
errors = []
records = []

total_start_time = time.time()
batch_times = []

with torch.no_grad():
    for inputs, paths in loader:
        inputs = inputs.to(device, non_blocking=True)
        batch_start = time.time()
        outputs = model(inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)

        batch_errors = compute_batch_reconstruction_errors(inputs, outputs)

        for path, err in zip(paths, batch_errors.cpu().numpy().tolist()):
            errors.append(err)
            records.append({
                "file_path": path,
                "reconstruction_error": float(err)
            })

total_end_time = time.time()
total_inference_time = total_end_time - total_start_time
avg_inference_time = total_inference_time / len(records)
images_per_second = len(records) / total_inference_time
avg_batch_time = np.mean(batch_times)

# =========================================================
# RESULTS
# =========================================================
errors_np = np.array(errors)
mean_error = float(np.mean(errors_np))
std_error = float(np.std(errors_np))
min_error = float(np.min(errors_np))
max_error = float(np.max(errors_np))

print(f"Mean error: {mean_error:.6f}")
print(f"Std error:  {std_error:.6f}")
print(f"Min error:  {min_error:.6f}")
print(f"Max error:  {max_error:.6f}")

save_json(records, os.path.join(RESULTS_DIR, "inference_records.json"))
save_json(
    {
        "num_images": len(records),
        "mean_error": mean_error,
        "std_error": std_error,
        "min_error": min_error,
        "max_error": max_error,
        "total_inference_time_per_image_sec": avg_inference_time,
        "images_per_second": images_per_second,
        "avg_batch_time_sec": float(avg_batch_time)
    },
    os.path.join(RESULTS_DIR, "inference_summary.json")
)

print("Inference finished.")