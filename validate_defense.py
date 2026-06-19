import pandas as pd
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Load analysis data and find high-confidence sample possessions
# ---------------------------------------------------------------------------
analysis_df = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'defensive_analysis.csv')
positions_df = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'player_positions_with_teams.csv')

# Group by possession and pick high-confidence ones
possession_groups = analysis_df.groupby('possession_id').first()

# Only consider possessions with a live-play classification
live_schemes = ['man', 'zone', 'variant']
live_possessions = possession_groups[possession_groups['possession_scheme'].isin(live_schemes)]

# Pick 6 samples: mix of highest confidence and spread across the game
# Sort by confidence, take top 6
sample_possessions = live_possessions.nlargest(6, 'possession_confidence')

print("Selected sample possessions:")
for pid, row in sample_possessions.iterrows():
    print(f"  Possession {pid:>3} | {row['possession_scheme']:>4} | "
          f"confidence={row['possession_confidence']:.3f} | "
          f"frames={int(row['possession_n_frames'])} | "
          f"mean_NOV={row['possession_mean_nov']:.0f}")

# ---------------------------------------------------------------------------
# 2. Extract representative frame from each possession
# ---------------------------------------------------------------------------
video_path = BASE_DIR / 'data' / 'videos' / 'game_trimmed.mp4'
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"ERROR: Could not open video at {video_path}")
    exit(1)

output_dir = BASE_DIR / 'data' / 'processed' / 'validation'
# Clear old sample images to prevent stale files from accumulating
if output_dir.exists():
    for old_img in output_dir.glob('sample_*.png'):
        old_img.unlink()
output_dir.mkdir(exist_ok=True)

# Team colors (BGR for OpenCV)
TEAM_COLORS = {
    1: (255, 100, 50),   # Blue-ish (Team 1)
    2: (50, 50, 255),    # Red-ish (Team 2)
    0: (180, 180, 180),  # Gray (Other)
}

for idx, (pid, row) in enumerate(sample_possessions.iterrows()):
    scheme = row['possession_scheme']
    confidence = row['possession_confidence']
    mean_nov = row['possession_mean_nov']
    n_frames = int(row['possession_n_frames'])
    
    # Get all frames in this possession
    poss_frames = analysis_df[analysis_df['possession_id'] == pid]
    
    # Pick the middle frame as the representative
    mid_idx = len(poss_frames) // 2
    rep_row = poss_frames.iloc[mid_idx]
    target_frame = int(rep_row['frame'])
    def_team = int(rep_row['def_team'])
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
    ret, frame = cap.read()
    if not ret:
        print(f"  Could not read frame {target_frame}")
        continue

    # Get player positions for this frame
    frame_players = positions_df[positions_df['frame'] == target_frame]
    
    # Separate on-court and off-court players
    if 'on_court' in frame_players.columns:
        on_court = frame_players[frame_players['on_court'] == True]
        off_court = frame_players[frame_players['on_court'] == False]
    else:
        on_court = frame_players
        off_court = pd.DataFrame()

    # Draw on-court players with bounding boxes
    for _, player in on_court.iterrows():
        team = int(player['team'])
        x, y = int(player['x']), int(player['y'])
        w, h = int(player['width']), int(player['height'])
        color = TEAM_COLORS.get(team, (180, 180, 180))

        # Draw bounding box
        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw team label
        label = f"T{team}"
        if team == def_team:
            label += " (DEF)"
        cv2.putText(frame, label, (x1, y1 - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw off-court (filtered) players with X markers
    for _, player in off_court.iterrows():
        cx, cy = int(player['x']), int(player['y'])
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 20, 2)

    # Draw classification banner at top
    banner_colors = {'primary': (50, 160, 50), 'man': (200, 120, 50), 'zone': (50, 100, 220),
                      'variant': (50, 150, 220), 'uncertain': (120, 120, 120)}
    banner_color = banner_colors.get(scheme, (120, 120, 120))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), banner_color, -1)

    text = (f"Poss {pid} | {scheme.upper()} | "
            f"NOV={mean_nov:.0f} | conf={confidence:.2f} | "
            f"{n_frames}fr | Def=T{def_team}")
    cv2.putText(frame, text, (10, 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Save
    filename = f"sample_{idx+1}_{scheme}_poss{pid}_frame{target_frame}.png"
    out_path = output_dir / filename
    cv2.imwrite(str(out_path), frame)
    print(f"  Saved: {out_path.name}")

cap.release()
print(f"\nAll validation frames saved to: {output_dir}")
print("\nPlease review each image and verify if the classification looks correct.")
print("Red X markers = filtered off-court detections (bench/coaches)")
