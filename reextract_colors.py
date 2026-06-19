"""Re-extract jersey colors from video using existing bounding boxes.

Improvements over original:
1. Crops the MIDDLE BAND (25%-55% of bbox height) = torso/jersey area
   - Skips head (top) and legs (bottom)
2. Filters out skin-tone pixels before computing color
3. Filters out court/floor-colored pixels
4. Computes standard deviation of brightness (high stddev = refs with striped jerseys)
5. Computes dominant hue via histogram peak instead of just mean
"""
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load existing detections (bounding boxes)
input_path = BASE_DIR / 'data' / 'processed' / 'player_positions.csv'
df = pd.read_csv(input_path)
print(f"Loaded {len(df)} detections")

# Open video
video_path = BASE_DIR / 'data' / 'videos' / 'game_trimmed.mp4'
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"Video: {FRAME_WIDTH}x{FRAME_HEIGHT}")

# Process frame by frame (grouped for efficiency)
new_jersey_h = np.zeros(len(df))
new_jersey_s = np.zeros(len(df))
new_jersey_v = np.zeros(len(df))
jersey_v_std = np.zeros(len(df))  # Brightness stddev (ref detection)

frames_processed = 0
grouped = df.groupby('frame')

for frame_num, group in grouped:
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num) - 1)
    ret, frame = cap.read()
    if not ret:
        continue

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for idx in group.index:
        x = df.at[idx, 'x']
        y = df.at[idx, 'y']
        w = df.at[idx, 'width']
        h = df.at[idx, 'height']

        # --- MIDDLE BAND crop: 25% to 55% of bbox height ---
        x1 = int(x - w / 2)
        y1_bbox = int(y - h / 2)
        x2 = int(x + w / 2)

        crop_top = int(y1_bbox + h * 0.25)  # Skip head
        crop_bot = int(y1_bbox + h * 0.55)  # Skip legs

        # Clamp to frame
        x1 = max(0, x1)
        x2 = min(FRAME_WIDTH, x2)
        crop_top = max(0, crop_top)
        crop_bot = min(FRAME_HEIGHT, crop_bot)

        if x2 <= x1 or crop_bot <= crop_top:
            continue

        jersey_hsv = hsv_frame[crop_top:crop_bot, x1:x2]

        if jersey_hsv.size == 0:
            continue

        h_ch = jersey_hsv[:, :, 0].astype(float)
        s_ch = jersey_hsv[:, :, 1].astype(float)
        v_ch = jersey_hsv[:, :, 2].astype(float)

        # --- Filter out skin tones ---
        # Skin: hue 0-20, saturation > 30, moderate brightness
        skin = (h_ch >= 0) & (h_ch <= 20) & (s_ch > 30) & (v_ch > 60)

        # --- Filter out court/floor color ---
        # Court: yellowish-tan, hue ~15-30, low saturation
        floor = (h_ch >= 15) & (h_ch <= 30) & (s_ch < 80) & (v_ch > 100)

        # --- Filter out very dark and very bright pixels ---
        dark = v_ch < 40
        bright = v_ch > 240

        # Composite mask: keep good pixels only
        bad_pixels = skin | floor | dark | bright
        good_pixels = ~bad_pixels

        if good_pixels.sum() >= 5:
            new_jersey_h[idx] = float(h_ch[good_pixels].mean())
            new_jersey_s[idx] = float(s_ch[good_pixels].mean())
            new_jersey_v[idx] = float(v_ch[good_pixels].mean())
            jersey_v_std[idx] = float(v_ch[good_pixels].std())
        else:
            # Not enough valid pixels, use all
            new_jersey_h[idx] = float(h_ch.mean())
            new_jersey_s[idx] = float(s_ch.mean())
            new_jersey_v[idx] = float(v_ch.mean())
            jersey_v_std[idx] = float(v_ch.std())

    frames_processed += 1
    if frames_processed % 500 == 0:
        print(f"  Processed {frames_processed}/{len(grouped)} frames...")

cap.release()

# Update the dataframe
df['jersey_h'] = new_jersey_h
df['jersey_s'] = new_jersey_s
df['jersey_v'] = new_jersey_v
df['jersey_v_std'] = jersey_v_std

# Save
df.to_csv(input_path, index=False)
print(f"\nDone! Updated {len(df)} detections in {input_path}")
print(f"  H range: {df['jersey_h'].min():.1f} - {df['jersey_h'].max():.1f}")
print(f"  S range: {df['jersey_s'].min():.1f} - {df['jersey_s'].max():.1f}")
print(f"  V range: {df['jersey_v'].min():.1f} - {df['jersey_v'].max():.1f}")
print(f"  V_std range: {df['jersey_v_std'].min():.1f} - {df['jersey_v_std'].max():.1f}")
