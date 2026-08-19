"""
generate_dataset.py
--------------------
Creates a SMALL SYNTHETIC multimodal healthcare dataset for 100 patients.

For every patient we create:
  - an image   (dataset/images/patient_XXX.jpg)
  - a signal   (dataset/signals/patient_XXX.csv)
  - a text note (stored directly inside metadata.csv)
  - an age
  - a target label (0 = Healthy, 1 = Disease)

The three modalities are made to be WEAKLY correlated with the target,
just like in a real (but simplified) medical problem:
  - Disease images  -> have a brighter/noisier "lesion" spot
  - Disease signals -> are noisier and have a higher frequency
  - Disease text    -> uses symptom-heavy phrases

This lets a model actually learn something from each modality,
instead of learning from pure noise.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------
np.random.seed(42)  # reproducibility: same "random" data every run

N_PATIENTS = 100
IMG_SIZE = 64          # 64x64 pixel images (kept small on purpose)
SIGNAL_LEN = 100        # 100 time points per signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
SIGNALS_DIR = os.path.join(DATASET_DIR, "signals")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SIGNALS_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Text templates
# ------------------------------------------------------------
healthy_texts = [
    "Patient reports no symptoms and feels well",
    "Routine checkup, patient appears healthy",
    "No complaints, normal energy levels",
    "Patient feels fine, no pain reported",
    "Regular visit, no abnormal symptoms noted",
    "Patient is active and reports good health",
    "No chest pain, no shortness of breath",
    "Patient sleeping well, no concerns raised",
]

disease_texts = [
    "Patient has mild chest discomfort",
    "Patient reports shortness of breath",
    "Complains of chest pain and fatigue",
    "Patient feels dizzy and weak",
    "Reports irregular heartbeat and tiredness",
    "Patient has persistent chest tightness",
    "Experiencing fatigue and mild chest pain",
    "Patient reports palpitations and discomfort",
]


def make_image(target: int, patient_seed: int) -> Image.Image:
    """
    Create a simple synthetic grayscale image.
    Healthy (0): a plain, mostly uniform circle.
    Disease (1): the circle has extra bright/noisy "spots" (like a lesion).
    """
    rng = np.random.RandomState(patient_seed)
    # start with a mid-gray background + small random noise texture
    base = 90 + rng.randint(-10, 10, size=(IMG_SIZE, IMG_SIZE))
    img_arr = np.clip(base, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    # draw a "body/organ" circle in the middle
    cx, cy, r = IMG_SIZE // 2, IMG_SIZE // 2, IMG_SIZE // 3
    circle_color = (140, 140, 150) if target == 0 else (150, 110, 110)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=circle_color)

    if target == 1:
        # disease patients get 2-4 small bright irregular spots (lesions)
        n_spots = rng.randint(2, 5)
        for _ in range(n_spots):
            sx = cx + rng.randint(-r + 5, r - 5)
            sy = cy + rng.randint(-r + 5, r - 5)
            sr = rng.randint(2, 5)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(230, 60, 60))
    else:
        # healthy patients get faint uniform texture only (no bright spots)
        n_spots = rng.randint(0, 2)
        for _ in range(n_spots):
            sx = cx + rng.randint(-r + 5, r - 5)
            sy = cy + rng.randint(-r + 5, r - 5)
            sr = rng.randint(1, 3)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(150, 150, 160))

    return img


def make_signal(target: int, patient_seed: int) -> pd.DataFrame:
    """
    Create a simple synthetic time-series signal (like a 1-lead heartbeat trace).
    Healthy (0): clean, low-frequency, low-noise sine wave.
    Disease (1): higher frequency + more noise + occasional irregular spikes.
    """
    rng = np.random.RandomState(patient_seed + 1000)
    t = np.linspace(0, 10, SIGNAL_LEN)  # time axis, 0 to 10 seconds

    if target == 0:
        freq = rng.uniform(0.8, 1.0)     # slow, regular rhythm
        noise_level = 0.03
        value = np.sin(2 * np.pi * freq * t) * 0.5 + 0.5
        value += rng.normal(0, noise_level, SIGNAL_LEN)
    else:
        freq = rng.uniform(1.3, 1.8)     # faster, irregular rhythm
        noise_level = 0.09
        value = np.sin(2 * np.pi * freq * t) * 0.5 + 0.5
        value += rng.normal(0, noise_level, SIGNAL_LEN)
        # add a couple of random irregular spikes (arrhythmia-like)
        n_spikes = rng.randint(1, 4)
        spike_idx = rng.choice(SIGNAL_LEN, size=n_spikes, replace=False)
        value[spike_idx] += rng.uniform(0.3, 0.6, size=n_spikes)

    value = np.clip(value, 0, 1.2)
    return pd.DataFrame({"time": np.round(t, 2), "value": np.round(value, 4)})


# ------------------------------------------------------------
# 2. Generate all patients
# ------------------------------------------------------------
records = []
targets = np.array([0] * (N_PATIENTS // 2) + [1] * (N_PATIENTS - N_PATIENTS // 2))
np.random.shuffle(targets)

for i in range(N_PATIENTS):
    patient_id = f"patient_{i+1:03d}"
    target = int(targets[i])
    seed = i  # unique seed per patient -> reproducible but different

    # --- image ---
    img = make_image(target, seed)
    img_filename = f"{patient_id}.jpg"
    img.save(os.path.join(IMAGES_DIR, img_filename))

    # --- signal ---
    signal_df = make_signal(target, seed)
    signal_filename = f"{patient_id}.csv"
    signal_df.to_csv(os.path.join(SIGNALS_DIR, signal_filename), index=False)

    # --- text ---
    rng = np.random.RandomState(seed + 2000)
    if target == 0:
        text = rng.choice(healthy_texts)
    else:
        text = rng.choice(disease_texts)

    # --- age ---
    age = int(np.random.RandomState(seed + 3000).randint(20, 81))

    records.append({
        "patient_id": patient_id,
        "image": f"images/{img_filename}",
        "signal": f"signals/{signal_filename}",
        "text": text,
        "age": age,
        "target": target,
    })

metadata = pd.DataFrame(records)
metadata.to_csv(os.path.join(DATASET_DIR, "metadata.csv"), index=False)

print(f"Created {N_PATIENTS} patients.")
print(metadata["target"].value_counts())
print(f"Saved to: {DATASET_DIR}")
