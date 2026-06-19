import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
input_path = BASE_DIR / 'data' / 'processed' / 'player_positions.csv'
df = pd.read_csv(input_path)

print(f"Total detections: {len(df)}")

# ---------------------------------------------------------------------------
# 2. Adaptive court filtering — IQR outlier removal per frame
# ---------------------------------------------------------------------------
# Removes bench players, coaches, and sideline detections BEFORE clustering.
# Uses IQR (interquartile range) on x/y positions — camera-agnostic.

def filter_court_players_adaptive(frame_df, iqr_multiplier=1.5):
    """Filter to on-court players using adaptive outlier removal.
    
    Uses IQR (Interquartile Range) method — robust to non-normal distributions.
    Players far from the main cluster are removed (bench, sideline, coaches).
    
    Args:
        frame_df: DataFrame with x, y columns for one frame
        iqr_multiplier: How many IQRs away = outlier (1.5-2.0 typical)
    
    Returns:
        Boolean mask of which detections to keep
    """
    if len(frame_df) < 5:
        return pd.Series(True, index=frame_df.index)
    
    x = frame_df['x'].values
    y = frame_df['y'].values
    
    # IQR-based outlier detection on both x and y
    q1_x, q3_x = np.percentile(x, [25, 75])
    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_x = q3_x - q1_x
    iqr_y = q3_y - q1_y
    
    # Ensure minimum IQR (don't filter too aggressively when players are clustered)
    iqr_x = max(iqr_x, 50)
    iqr_y = max(iqr_y, 30)
    
    x_low = q1_x - iqr_multiplier * iqr_x
    x_high = q3_x + iqr_multiplier * iqr_x
    y_low = q1_y - iqr_multiplier * iqr_y
    y_high = q3_y + iqr_multiplier * iqr_y
    
    keep = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    return pd.Series(keep, index=frame_df.index)


print("Applying adaptive court filtering (IQR)...")
df['on_court'] = False

total_removed = 0
for frame_num, group in df.groupby('frame'):
    keep_mask = filter_court_players_adaptive(group)
    df.loc[keep_mask[keep_mask].index, 'on_court'] = True
    total_removed += (~keep_mask).sum()

on_court_count = df['on_court'].sum()
print(f"  On-court: {on_court_count} | Off-court (filtered): {total_removed}")
print(f"  Avg removed per frame: {total_removed / df['frame'].nunique():.1f}")

# ---------------------------------------------------------------------------
# 3. Pre-filter: detect refs by brightness standard deviation
# ---------------------------------------------------------------------------
# Refs wear striped black-and-white jerseys → high brightness stddev
# Also filter out detections with very low stddev (likely court/background)
has_std = 'jersey_v_std' in df.columns

if has_std:
    ref_threshold = df['jersey_v_std'].quantile(0.90)  # Top 10% stddev = likely refs
    ref_mask = df['jersey_v_std'] > ref_threshold
    print(f"Ref detection: {ref_mask.sum()} detections with high brightness variance "
          f"(stddev > {ref_threshold:.1f})")
else:
    ref_mask = pd.Series(False, index=df.index)
    print("No jersey_v_std column — skipping ref detection. Run reextract_colors.py first.")

# ---------------------------------------------------------------------------
# 4. Prepare features — cluster only on-court, non-ref detections
# ---------------------------------------------------------------------------
# Weight hue heavily since it distinguishes jersey colors best
HUE_WEIGHT = 3.0

h = df['jersey_h'].values
s = df['jersey_s'].values
v = df['jersey_v'].values

# --- Pre-filter: remove low-saturation detections before clustering ---
# Very low saturation means the detected "jersey" color is essentially gray/brown
# (court floor, washed-out background, shadowed areas). These pollute the clusters
# by inflating the low-hue team count.
MIN_JERSEY_SATURATION = 30
low_sat_mask = s < MIN_JERSEY_SATURATION
low_sat_count = (low_sat_mask & df['on_court'].values & ~ref_mask).sum()
print(f"Low-saturation pre-filter: {low_sat_count} on-court detections with S<{MIN_JERSEY_SATURATION} excluded from clustering")

color_features = np.column_stack([h * HUE_WEIGHT, s, v])
valid_mask = (~np.isnan(color_features).any(axis=1) & ~ref_mask
              & df['on_court'].values & ~low_sat_mask)

print(f"Clustering on {valid_mask.sum()} on-court, non-ref, colored detections")

# ---------------------------------------------------------------------------
# 5. K-Means (k=2 for two teams, refs/off-court already filtered)
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=7, n_init=10)
labels = np.full(len(df), -1)
labels[valid_mask] = kmeans.fit_predict(color_features[valid_mask])

# Assign team labels: larger cluster = Team 1, smaller = Team 2
cluster_sizes = pd.Series(labels[valid_mask]).value_counts().sort_values(ascending=False)
team_map = {cluster_sizes.index[0]: 1, cluster_sizes.index[1]: 2}

# Map labels to team IDs; refs and off-court get team 0
df['team'] = 0  # Default: other/ref/off-court
for idx in range(len(df)):
    if not df.iloc[idx]['on_court']:
        df.at[df.index[idx], 'team'] = 0  # Off-court
    elif ref_mask.iloc[idx]:
        df.at[df.index[idx], 'team'] = 0  # Ref
    elif labels[idx] >= 0:
        df.at[df.index[idx], 'team'] = team_map.get(labels[idx], 0)

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print("\n--- Cluster Summary ---")
for cluster_id in range(2):
    center = kmeans.cluster_centers_[cluster_id]
    team_label = team_map.get(cluster_id, 0)
    count = (labels == cluster_id).sum()
    display_h = center[0] / HUE_WEIGHT
    print(f"  Team {team_label}: {count} detections | "
          f"Avg HSV = ({display_h:.1f}, {center[1]:.1f}, {center[2]:.1f})")
print(f"  Refs/Other: {ref_mask.sum()} detections (filtered by brightness stddev)")
print(f"  Off-court: {total_removed} detections (filtered by IQR)")

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
output_path = BASE_DIR / 'data' / 'processed' / 'player_positions_with_teams.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

# Per-frame team counts (on-court players only)
on_court_df = df[(df['team'] > 0) & (df['on_court'])]
team_counts = on_court_df.groupby(['frame', 'team']).size().unstack(fill_value=0)
if 1 in team_counts.columns:
    print(f"\nAverage Team 1 players per frame (on-court): {team_counts[1].mean():.1f}")
if 2 in team_counts.columns:
    print(f"Average Team 2 players per frame (on-court): {team_counts[2].mean():.1f}")

# ---------------------------------------------------------------------------
# 8. Visualization
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors_map = {0: 'gray', 1: 'blue', 2: 'red'}
point_colors = [colors_map.get(t, 'gray') for t in df['team']]

axes[0].scatter(df['jersey_h'], df['jersey_s'], c=point_colors, alpha=0.3, s=5)
axes[0].set_xlabel('Hue')
axes[0].set_ylabel('Saturation')
axes[0].set_title('Jersey Colors: Hue vs Saturation')

axes[1].scatter(df['jersey_h'], df['jersey_v'], c=point_colors, alpha=0.3, s=5)
axes[1].set_xlabel('Hue')
axes[1].set_ylabel('Value (Brightness)')
axes[1].set_title('Jersey Colors: Hue vs Brightness')

plt.suptitle('Team Classification by Jersey Color\n(Blue=Team 1, Red=Team 2, Gray=Refs/Off-court)')
plt.tight_layout()
plt.savefig(str(BASE_DIR / 'data' / 'processed' / 'team_clusters.png'), dpi=150)
plt.show()
print("Cluster visualization saved")
