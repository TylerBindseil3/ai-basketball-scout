import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import ConvexHull
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Load data and get video dimensions
# ---------------------------------------------------------------------------
video_path = BASE_DIR / 'data' / 'videos' / 'game_trimmed.mp4'
cap = cv2.VideoCapture(str(video_path))
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT

input_path = BASE_DIR / 'data' / 'processed' / 'player_positions_with_teams.csv'
df = pd.read_csv(input_path)

# Only consider on-court team players (team 1 and 2)
# The on_court column was set by classify_teams.py's IQR filter
if 'on_court' in df.columns:
    before = len(df)
    df = df[(df['team'].isin([1, 2])) & (df['on_court'] == True)].copy()
    print(f"Filtered to on-court team players: {len(df)} (from {before} total)")
else:
    # Fallback if on_court column missing (old data)
    df = df[df['team'].isin([1, 2])].copy()
    print(f"Warning: no on_court column found — using all team players ({len(df)})")

MIN_PLAYERS_PER_TEAM = 4
# Minimum total on-court players (both teams combined) to analyze a frame
# In a real game there should be 10; fewer suggests bad detections or dead ball
MIN_TOTAL_PLAYERS = 8
# Maximum ratio of team sizes — if one team has 3x+ more players, likely dead ball
MAX_TEAM_RATIO = 3.0
# Frame-relative minimum hull area (2% of frame area)
MIN_HULL_FRACTION = 0.02
MIN_ALL_PLAYERS_HULL = MIN_HULL_FRACTION * FRAME_AREA
# Maximum x-range spread ratio for half-court play
MAX_X_SPREAD_RATIO = 0.65
# Free throw detection: if player bounding region is much taller than wide,
# players are lined up along the lane (free throw formation)
MAX_FREE_THROW_ASPECT = 0.6  # x_range / y_range — below this = free throw

print(f"  Min hull area: {MIN_ALL_PLAYERS_HULL:.0f} px² ({MIN_HULL_FRACTION*100:.0f}% of frame)")
print(f"  Max x-spread ratio: {MAX_X_SPREAD_RATIO}")
print(f"  Min total players: {MIN_TOTAL_PLAYERS}")

# ---------------------------------------------------------------------------
# 2. Per-frame metrics
# ---------------------------------------------------------------------------

def convex_hull_area(points):
    """Compute convex hull area. Returns 0 if < 3 points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, .volume gives area
    except Exception:
        return 0.0


def compute_frame_metrics(frame_df, frame_width):
    """Compute defensive scheme metrics for a single frame.
    
    Returns a dict with metrics, or None if not enough players or dead ball.
    """
    t1 = frame_df[frame_df['team'] == 1][['x', 'y']].values
    t2 = frame_df[frame_df['team'] == 2][['x', 'y']].values

    if len(t1) < MIN_PLAYERS_PER_TEAM or len(t2) < MIN_PLAYERS_PER_TEAM:
        return None

    # --- Minimum total players check ---
    total_players = len(t1) + len(t2)
    if total_players < MIN_TOTAL_PLAYERS:
        return None  # Not enough players detected — likely bad frame

    # --- Team balance check ---
    # In live play, teams should be roughly equal (5v5).
    # If one team has 3x+ more detections, it's likely a dead ball.
    ratio = max(len(t1), len(t2)) / max(min(len(t1), len(t2)), 1)
    if ratio > MAX_TEAM_RATIO:
        return None

    # --- Filter: X-range spread ratio (frame-relative) ---
    all_positions = np.vstack([t1, t2])
    x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
    y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
    x_spread_ratio = x_range / frame_width
    if x_spread_ratio > MAX_X_SPREAD_RATIO:
        return None  # Players span too much of the frame — transition/dead ball

    # --- Filter: Free throw detection (aspect ratio) ---
    # During free throws, players line up along the lane — narrow x, tall y.
    # The aspect ratio (x_range / y_range) will be very low.
    if y_range > 0:
        aspect_ratio = x_range / y_range
        if aspect_ratio < MAX_FREE_THROW_ASPECT:
            return None  # Players in narrow vertical band — likely free throw

    # --- Filter: Minimum spatial spread (dead ball check, frame-relative) ---
    all_hull_area = convex_hull_area(all_positions)
    if all_hull_area < MIN_ALL_PLAYERS_HULL:
        return None  # Players too clustered — likely dead ball / huddle

    # --- Filter: Team spatial separation ---
    # In live play, the two teams should have some spatial separation.
    # If both team centroids are nearly identical, it's likely a dead ball.
    centroid_1 = t1.mean(axis=0)
    centroid_2 = t2.mean(axis=0)
    centroid_distance = np.linalg.norm(centroid_1 - centroid_2)
    if centroid_distance < 15:  # Teams on top of each other = dead ball
        return None

    # Convex hull area per team
    hull_area_1 = convex_hull_area(t1)
    hull_area_2 = convex_hull_area(t2)

    # Infer defensive team: smaller hull = more compact = likely defense
    if hull_area_1 < hull_area_2:
        def_team, off_team = 1, 2
        def_positions, off_positions = t1, t2
        def_hull, off_hull = hull_area_1, hull_area_2
    else:
        def_team, off_team = 2, 1
        def_positions, off_positions = t2, t1
        def_hull, off_hull = hull_area_2, hull_area_1

    # --- Metric 1: Nearest-opponent distance variance ---
    cross_distances = cdist(def_positions, off_positions)
    nearest_opp_distances = cross_distances.min(axis=1)
    nearest_opp_variance = float(nearest_opp_distances.var())
    nearest_opp_mean = float(nearest_opp_distances.mean())

    # --- Metric 2: Within-team pairwise spacing ratio ---
    def_pairwise = pdist(def_positions)
    spacing_ratio = float(def_pairwise.std() / def_pairwise.mean()) if def_pairwise.mean() > 0 else 0.0

    # --- Metric 3: Spread ratio (defense hull / offense hull) ---
    spread_ratio = def_hull / off_hull if off_hull > 0 else 0.0

    return {
        'def_team': def_team,
        'off_team': off_team,
        'nearest_opp_variance': nearest_opp_variance,
        'nearest_opp_mean': nearest_opp_mean,
        'spacing_ratio': spacing_ratio,
        'def_hull_area': def_hull,
        'off_hull_area': off_hull,
        'spread_ratio': spread_ratio,
        'all_hull_area': all_hull_area,
        'x_spread_ratio': x_spread_ratio,
        'centroid_dist': centroid_distance,
        'all_centroid_x': float(all_positions[:, 0].mean()),
        'all_centroid_y': float(all_positions[:, 1].mean()),
        'n_def': len(def_positions),
        'n_off': len(off_positions),
    }


print("Computing per-frame metrics...")
frames = sorted(df['frame'].unique())
results = []

for frame_num in frames:
    frame_df = df[df['frame'] == frame_num]
    metrics = compute_frame_metrics(frame_df, FRAME_WIDTH)
    if metrics is not None:
        metrics['frame'] = frame_num
        results.append(metrics)

metrics_df = pd.DataFrame(results)
skipped = len(frames) - len(metrics_df)
print(f"  Analyzed {len(metrics_df)} frames (of {len(frames)} total)")
print(f"  Skipped {skipped} frames (insufficient players, dead ball, or transition)")

# ---------------------------------------------------------------------------
# 3. Possession detection by temporal gaps + defensive team switches
# ---------------------------------------------------------------------------

def detect_possessions_by_gaps(df, gap_threshold=5, switch_threshold=3):
    """Detect possession boundaries using temporal gaps and def-team switches.
    
    Args:
        df: DataFrame with 'frame' and 'def_team' columns, sorted by frame
        gap_threshold: Number of sampled frames of gap to split possessions
        switch_threshold: Number of consecutive def-team switches to split
    
    Returns:
        Array of possession IDs aligned with df index
    """
    if len(df) == 0:
        return np.array([])
    
    possession_ids = np.zeros(len(df), dtype=int)
    current_possession = 0
    possession_ids[0] = current_possession
    
    frames = df['frame'].values
    def_teams = df['def_team'].values
    
    # Track consecutive switches for secondary signal
    consecutive_switches = 0
    
    for i in range(1, len(df)):
        # Primary: temporal gap
        frame_gap = frames[i] - frames[i - 1]
        # Convert raw frame gap to sampled frame gap (we sample every 10 frames)
        sampled_gap = frame_gap / 10
        
        if sampled_gap > gap_threshold:
            current_possession += 1
            consecutive_switches = 0
        # Secondary: defensive team switch streak
        elif def_teams[i] != def_teams[i - 1]:
            consecutive_switches += 1
            if consecutive_switches >= switch_threshold:
                current_possession += 1
                consecutive_switches = 0
        else:
            consecutive_switches = 0
        
        possession_ids[i] = current_possession
    
    return possession_ids


print("Detecting possessions...")
metrics_df = metrics_df.sort_values('frame').reset_index(drop=True)
metrics_df['possession_id'] = detect_possessions_by_gaps(metrics_df)

n_possessions = metrics_df['possession_id'].nunique()
print(f"  Detected {n_possessions} possessions")

# ---------------------------------------------------------------------------
# 4. Possession-level aggregation and classification
# ---------------------------------------------------------------------------

MIN_POSSESSION_FRAMES = 3  # Minimum frames for a confident classification

def aggregate_and_classify_possessions(df):
    """Aggregate per-frame metrics into possession-level metrics and classify.
    
    Returns the df with added possession-level columns.
    """
    possession_stats = []
    
    for pid, group in df.groupby('possession_id'):
        hull_std = group['all_hull_area'].std() if len(group) > 1 else 0.0
        hull_mean = group['all_hull_area'].mean()
        # Hull coefficient of variation: low = players aren't moving = dead ball
        hull_cv = hull_std / hull_mean if hull_mean > 0 else 0.0
        
        # Centroid movement: how much the overall player centroid moves across frames
        # Low movement = players standing still = dead ball
        centroid_x_std = group['all_centroid_x'].std() if len(group) > 1 else 0.0
        centroid_y_std = group['all_centroid_y'].std() if len(group) > 1 else 0.0
        centroid_movement = np.sqrt(centroid_x_std**2 + centroid_y_std**2)
        
        stats = {
            'possession_id': pid,
            'possession_n_frames': len(group),
            'possession_mean_nov': group['nearest_opp_variance'].mean(),
            'possession_mean_nopp_dist': group['nearest_opp_mean'].mean(),
            'possession_mean_spread': group['spread_ratio'].mean(),
            'possession_nov_std': group['nearest_opp_variance'].std() if len(group) > 1 else 0.0,
            'possession_hull_cv': hull_cv,
            'possession_centroid_movement': centroid_movement,
            # X-spread consistency: low std = players maintain same lateral arrangement
            # Free throws: players stay in the lane, x_spread barely changes
            # Live play: offense moves the ball, x_spread changes significantly
            'possession_xspread_std': group['x_spread_ratio'].std() if len(group) > 1 else 0.0,
        }
        possession_stats.append(stats)
    
    poss_df = pd.DataFrame(possession_stats)
    
    # --- Dead ball detection (multi-signal) ---
    # Signal 1: Static hull — hull area barely changes between frames
    STATIC_HULL_CV_THRESHOLD = 0.15  # 15% variation threshold
    static_hull = poss_df['possession_hull_cv'] < STATIC_HULL_CV_THRESHOLD
    
    # Signal 2: Low centroid movement — overall player group barely moves
    STATIC_CENTROID_THRESHOLD = 20.0  # pixels of centroid std
    static_centroid = poss_df['possession_centroid_movement'] < STATIC_CENTROID_THRESHOLD
    
    # Signal 3: Low NOV with low centroid movement
    NOV_DEAD_BALL_FLOOR = 350
    
    # Signal 4: Low x-spread consistency — players maintain same lateral arrangement
    XSPREAD_STATIC_THRESHOLD = 0.03
    static_xspread = poss_df['possession_xspread_std'] < XSPREAD_STATIC_THRESHOLD
    
    # Signal 5: High spread ratio + low NOV — both teams in same area
    # During actual man-to-man, defense is compact (spread_ratio < 0.3)
    # During free throws/dead balls, both teams cluster together (spread_ratio > 0.4)
    HIGH_SPREAD_THRESHOLD = 0.4
    high_spread = poss_df['possession_mean_spread'] > HIGH_SPREAD_THRESHOLD
    
    # A possession is dead ball if ANY of:
    #   1. hull is static (free throw / timeout)
    #   2. centroid barely moves AND NOV is low
    #   3. x-spread is static AND NOV is low
    #   4. spread ratio is high AND NOV is low (both teams in same area = dead ball)
    dead_ball_mask = (
        static_hull |
        (static_centroid & (poss_df['possession_mean_nov'] < NOV_DEAD_BALL_FLOOR)) |
        (static_xspread & (poss_df['possession_mean_nov'] < NOV_DEAD_BALL_FLOOR)) |
        (high_spread & (poss_df['possession_mean_nov'] < NOV_DEAD_BALL_FLOOR))
    )
    
    n_dead = dead_ball_mask.sum()
    if n_dead > 0:
        print(f"  Dead ball possessions detected: {n_dead}")
        print(f"    Static hull (CV<{STATIC_HULL_CV_THRESHOLD}): {static_hull.sum()}")
        n_centroid_nov = (static_centroid & (poss_df['possession_mean_nov'] < NOV_DEAD_BALL_FLOOR) & ~static_hull).sum()
        print(f"    Low centroid + low NOV: {n_centroid_nov}")
        n_xspread = (static_xspread & (poss_df['possession_mean_nov'] < NOV_DEAD_BALL_FLOOR) & ~static_hull & ~static_centroid).sum()
        print(f"    Static x-spread + low NOV: {n_xspread}")
        n_spread = (high_spread & (poss_df['possession_mean_nov'] < NOV_DEAD_BALL_FLOOR) & ~static_hull & ~static_centroid & ~static_xspread).sum()
        print(f"    High spread + low NOV: {n_spread}")
    
    # -----------------------------------------------------------------------
    # Classification: Determine if there are 1 or 2 defensive schemes
    # -----------------------------------------------------------------------
    # Instead of using median (which always produces 50/50), use Gaussian
    # Mixture Model to determine if the data actually has 2 clusters.
    # If unimodal → report dominant scheme. If bimodal → use the natural gap.
    #
    # Key insight from basketball analytics research:
    #  - Man-to-man: low NOV (defenders track specific players → consistent distances)
    #  - Zone: high NOV (defenders guard areas → variable distances to nearest opponent)
    #  - Spacing ratio also helps: zone defenders are more evenly spaced
    
    from sklearn.mixture import GaussianMixture
    
    poss_df['possession_scheme'] = 'uncertain'
    poss_df.loc[dead_ball_mask, 'possession_scheme'] = 'dead_ball'
    
    classifiable = (
        (poss_df['possession_n_frames'] >= MIN_POSSESSION_FRAMES) &
        (~dead_ball_mask)
    )
    confident_poss = poss_df[classifiable]
    
    if len(confident_poss) < 4:
        # Not enough data to classify
        print(f"  Too few classifiable possessions ({len(confident_poss)}) for scheme detection")
        return poss_df, None
    
    # Use NOV as the primary classification feature
    nov_values = confident_poss['possession_mean_nov'].values.reshape(-1, 1)
    
    # Fit 1-component and 2-component GMMs and compare with BIC
    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(nov_values)
    gmm2 = GaussianMixture(n_components=2, random_state=42).fit(nov_values)
    
    bic1 = gmm1.bic(nov_values)
    bic2 = gmm2.bic(nov_values)
    
    # BIC difference: positive = 1-component is better (unimodal)
    bic_diff = bic2 - bic1
    # Require meaningful BIC improvement (> 10 is "very strong" evidence per Kass & Raftery)
    BIMODAL_BIC_THRESHOLD = -10
    
    is_bimodal = bic_diff < BIMODAL_BIC_THRESHOLD
    
    print(f"  GMM analysis: BIC(1-component)={bic1:.1f}, BIC(2-component)={bic2:.1f}")
    print(f"  BIC difference: {bic_diff:.1f} ({'bimodal — 2 schemes detected' if is_bimodal else 'unimodal — 1 dominant scheme'})")
    
    threshold = None
    
    if is_bimodal:
        # Two distinct clusters exist — use the GMM boundary
        means = gmm2.means_.flatten()
        
        # The threshold is where the posterior probabilities cross
        threshold = (means[0] + means[1]) / 2  # Simplified; GMM posterior is more precise
        
        poss_df.loc[classifiable, 'possession_scheme'] = confident_poss['possession_mean_nov'].apply(
            lambda x: 'man' if x < threshold else 'zone'
        )
        
        man_count = (poss_df['possession_scheme'] == 'man').sum()
        zone_count = (poss_df['possession_scheme'] == 'zone').sum()
        print(f"  Threshold (GMM boundary): {threshold:.1f}")
        print(f"  Man: {man_count} possessions, Zone: {zone_count} possessions")
        
        # Confidence = posterior probability of assigned class
        probs = gmm2.predict_proba(nov_values)
        conf_values = probs.max(axis=1)
        confident_poss_copy = confident_poss.copy()
        confident_poss_copy['_gmm_conf'] = conf_values
        poss_df.loc[classifiable, 'possession_confidence'] = confident_poss_copy['_gmm_conf'].values
        
    else:
        # Unimodal — team plays predominantly one scheme
        # Determine which scheme based on the overall NOV level
        overall_mean = float(nov_values.mean())
        overall_std = float(nov_values.std())
        
        # Unimodal: team plays one consistent scheme throughout the game.
        # Use nearest-opponent distance to infer scheme type:
        #  - Tight proximity (avg < 15% of frame width) → man-to-man
        #    (defenders tracking specific players stay close)
        #  - Loose proximity → zone (defenders guard areas, farther from attackers)
        avg_nopp = float(confident_poss['possession_mean_nopp_dist'].mean())
        nopp_threshold = FRAME_WIDTH * 0.15  # ~96px at 640 width
        
        if avg_nopp < nopp_threshold:
            dominant_scheme = 'man'
            scheme_name = 'Man-to-Man'
        else:
            dominant_scheme = 'zone'
            scheme_name = 'Zone'
        
        print(f"  Scheme: {scheme_name} (avg nearest-opp dist={avg_nopp:.1f}px, threshold={nopp_threshold:.0f}px)")
        print(f"  {len(confident_poss)} possessions, mean NOV={overall_mean:.1f}, std={overall_std:.1f}")
        
        # Classify all possessions as the dominant scheme
        poss_df.loc[classifiable, 'possession_scheme'] = dominant_scheme
        
        # Flag statistical outliers (> 2 std from mean) as 'variant'
        # These represent possessions that genuinely look different
        outlier_threshold_high = overall_mean + 2 * overall_std
        outlier_mask = classifiable & (poss_df['possession_mean_nov'] > outlier_threshold_high)
        poss_df.loc[outlier_mask, 'possession_scheme'] = 'variant'
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            print(f"  Variant possessions (NOV > {outlier_threshold_high:.0f}): {outlier_count}")
        
        # Confidence based on distance from overall mean (normalized)
        poss_df.loc[classifiable, 'possession_confidence'] = 1.0 - abs(
            poss_df.loc[classifiable, 'possession_mean_nov'] - overall_mean
        ) / max(overall_mean, 1.0)
        poss_df.loc[classifiable, 'possession_confidence'] = poss_df.loc[classifiable, 'possession_confidence'].clip(0, 1)
    
    # Mark uncertain and dead_ball possessions with 0 confidence
    poss_df.loc[
        poss_df['possession_scheme'].isin(['uncertain', 'dead_ball']),
        'possession_confidence'
    ] = 0.0
    
    return poss_df, threshold


print("Classifying possessions...")
poss_df, threshold_used = aggregate_and_classify_possessions(metrics_df)

# Merge possession-level classification back to per-frame data
metrics_df = metrics_df.merge(
    poss_df[['possession_id', 'possession_scheme', 'possession_confidence',
             'possession_n_frames', 'possession_mean_nov', 'possession_mean_nopp_dist',
             'possession_nov_std', 'possession_hull_cv', 'possession_centroid_movement']],
    on='possession_id',
    how='left'
)

# Use possession scheme as the final scheme label
metrics_df['scheme'] = metrics_df['possession_scheme']

# ---------------------------------------------------------------------------
# 5. Save per-frame results
# ---------------------------------------------------------------------------
output_path = BASE_DIR / 'data' / 'processed' / 'defensive_analysis.csv'
metrics_df.to_csv(output_path, index=False)
print(f"\nPer-frame results saved to: {output_path}")

# ---------------------------------------------------------------------------
# 6. Game summary
# ---------------------------------------------------------------------------
total_frames = len(metrics_df)
man_frames = (metrics_df['scheme'] == 'man').sum()
zone_frames = (metrics_df['scheme'] == 'zone').sum()
variant_frames = (metrics_df['scheme'] == 'variant').sum()
dead_ball_frames = (metrics_df['scheme'] == 'dead_ball').sum()

# Possession-level counts
man_poss = (poss_df['possession_scheme'] == 'man').sum()
zone_poss = (poss_df['possession_scheme'] == 'zone').sum()
variant_poss = (poss_df['possession_scheme'] == 'variant').sum()
uncertain_poss = (poss_df['possession_scheme'] == 'uncertain').sum()
dead_ball_poss = (poss_df['possession_scheme'] == 'dead_ball').sum()

# Count live play frames
live_frames = man_frames + zone_frames + variant_frames
live_poss = man_poss + zone_poss + variant_poss

print("\n" + "=" * 55)
print("GAME SUMMARY — DEFENSIVE SCHEME ANALYSIS")
print("=" * 55)
print(f"  Frames analyzed:       {total_frames}")
print(f"  Live play frames:      {live_frames} ({100 * live_frames / max(total_frames, 1):.1f}%)")
print(f"  Dead ball frames:      {dead_ball_frames} ({100 * dead_ball_frames / max(total_frames, 1):.1f}%)")
print()
print(f"  Possessions:           {n_possessions}")
print(f"    Live play:           {live_poss}")
if man_poss > 0:
    print(f"      Man-to-man:        {man_poss}")
if zone_poss > 0:
    print(f"      Zone:              {zone_poss}")
if variant_poss > 0:
    print(f"      Variant:           {variant_poss}")
print(f"    Uncertain (<{MIN_POSSESSION_FRAMES} frames): {uncertain_poss}")
print(f"    Dead ball (static):  {dead_ball_poss}")
print()
print("  Metric Averages (defensive team, live play):")
live_df = metrics_df[metrics_df['scheme'].isin(['man', 'zone', 'variant'])]
if len(live_df) > 0:
    print(f"    Nearest-opp distance (mean):     {live_df['nearest_opp_mean'].mean():.1f} px")
    print(f"    Nearest-opp distance (variance): {live_df['nearest_opp_variance'].mean():.1f}")
    print(f"    Within-team spacing ratio:       {live_df['spacing_ratio'].mean():.3f}")
    print(f"    Hull area ratio (def/off):       {live_df['spread_ratio'].mean():.3f}")
print()

# Breakdown by scheme (only show categories that have data)
for scheme in ['man', 'zone', 'variant']:
    subset = metrics_df[metrics_df['scheme'] == scheme]
    if len(subset) > 0:
        label = {'man': 'MAN-TO-MAN', 'zone': 'ZONE', 'variant': 'VARIANT'}.get(scheme, scheme.upper())
        print(f"  {label} averages:")
        print(f"    Nearest-opp variance:  {subset['nearest_opp_variance'].mean():.1f}")
        print(f"    Spacing ratio:         {subset['spacing_ratio'].mean():.3f}")
        print(f"    Hull area ratio:       {subset['spread_ratio'].mean():.3f}")
        print()

# Diagnostic: show all classified possessions with metrics
live_poss_detail = poss_df[poss_df['possession_scheme'].isin(['man', 'zone', 'variant'])].sort_values('possession_mean_nov')
if len(live_poss_detail) > 0:
    print("  All classified possessions (sorted by NOV):")
    print(f"  {'PID':>4} {'Scheme':>8} {'Frames':>6} {'NOV':>8} {'Conf':>6} {'HullCV':>8} {'CtrMov':>8} {'NoppDist':>8}")
    for _, row in live_poss_detail.iterrows():
        print(f"  {int(row['possession_id']):>4} {row['possession_scheme']:>8} {int(row['possession_n_frames']):>6} {row['possession_mean_nov']:>8.1f} {row['possession_confidence']:>6.3f} {row['possession_hull_cv']:>8.3f} {row['possession_centroid_movement']:>8.1f} {row['possession_mean_nopp_dist']:>8.1f}")
    print()

# ---------------------------------------------------------------------------
# 7. Visualizations
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Plot 1: Scheme classification over time (possession-level)
scheme_map = {'man': 0, 'zone': 1, 'variant': 0.75, 'uncertain': 0.5, 'dead_ball': -0.5}
scheme_numeric = metrics_df['scheme'].map(scheme_map)
color_map = {'man': '#2196F3', 'zone': '#FF5722',
             'variant': '#FF9800', 'uncertain': '#9E9E9E', 'dead_ball': '#795548'}
colors = [color_map.get(s, '#9E9E9E') for s in metrics_df['scheme']]
axes[0].scatter(metrics_df['frame'], scheme_numeric, c=colors, s=8, alpha=0.7)
axes[0].set_yticks([-0.5, 0, 0.5, 0.75, 1])
axes[0].set_yticklabels(['Dead Ball', 'Man-to-Man', 'Uncertain', 'Variant', 'Zone'])
axes[0].set_title('Defensive Scheme Classification Over Time (Possession-Level)')
axes[0].set_ylabel('Scheme')

# Plot 2: Nearest-opponent distance variance over time
axes[1].plot(metrics_df['frame'], metrics_df['nearest_opp_variance'],
             color='#666', alpha=0.3, linewidth=0.5)
# Rolling average for readability
window = min(20, len(metrics_df) // 5) if len(metrics_df) > 10 else 3
rolling_nov = metrics_df['nearest_opp_variance'].rolling(window, center=True).mean()
axes[1].plot(metrics_df['frame'], rolling_nov, color='#9C27B0', linewidth=1.5,
             label=f'{window}-frame rolling avg')
if threshold_used is not None:
    axes[1].axhline(y=threshold_used, color='red', linestyle='--', alpha=0.5, label='GMM threshold')
axes[1].set_ylabel('Nearest-Opp Variance')
axes[1].set_title('Nearest-Opponent Distance Variance (key metric)')
axes[1].legend()

# Plot 3: Spread ratio over time
axes[2].plot(metrics_df['frame'], metrics_df['spread_ratio'],
             color='#666', alpha=0.3, linewidth=0.5)
rolling_sr = metrics_df['spread_ratio'].rolling(window, center=True).mean()
axes[2].plot(metrics_df['frame'], rolling_sr, color='#009688', linewidth=1.5,
             label=f'{window}-frame rolling avg')
axes[2].set_ylabel('Def/Off Hull Ratio')
axes[2].set_xlabel('Frame')
axes[2].set_title('Defensive Spread Ratio (def hull / off hull)')
axes[2].legend()

plt.tight_layout()
plot_path = BASE_DIR / 'data' / 'processed' / 'defensive_analysis.png'
plt.savefig(str(plot_path), dpi=150)
print(f"Plots saved to: {plot_path}")
