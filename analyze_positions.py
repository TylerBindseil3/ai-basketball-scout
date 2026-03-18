import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

df = pd.read_csv('data/processed/player_positions.csv')

# Get detections from one frame (pick a frame number from your data)
# Filter to court area
court_players = df[(df['y'] >= 80) & (df['y'] <= 200)]

# Check average detections per frame now
detections_per_frame = court_players.groupby('frame').size()
print(f"Average players per frame: {detections_per_frame.mean():.1f}")
print(f"Min: {detections_per_frame.min()}")
print(f"Max: {detections_per_frame.max()}")

# Look at the y-value distribution
print("Y values - Min:", df['y'].min())
print("Y values - Max:", df['y'].max())
print("Y values - Mean:", df['y'].mean())

# Initializing results list so we can append dictionaries with frame and ratio values for each frame
results = []

for frame in court_players['frame'].unique():
    frame_players = court_players[court_players['frame'] == frame]
    
    if len(frame_players) < 2:
        continue
    
    coordinates = frame_players[['x', 'y']].values
    distances = pdist(coordinates)
    
    ratio = distances.std() / distances.mean()
    
    results.append({
        'frame': frame,
        'ratio': ratio
    })

# After the loop
print(f"Total frames analyzed: {len(results)}")
print(f"First few results: {results[:5]}")

# Calculating pairwise distances between players
distances = pdist(coordinates)

results_df = pd.DataFrame(results)

plt.xlabel('Frame')
plt.ylabel('Distance Ratio')
plt.title('Player Position Variability Over Time')
plt.plot(results_df['frame'], results_df['ratio'], marker='o')
plt.grid()
plt.show()