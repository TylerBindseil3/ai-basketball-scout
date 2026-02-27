import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/player_positions.csv')

# Get detections from one frame (pick a frame number from your data)
# Filter to court area
court_players = df[(df['y'] >= 200) & (df['y'] <= 350)]

# Check average detections per frame now
detections_per_frame = court_players.groupby('frame').size()
print(f"Average players per frame: {detections_per_frame.mean():.1f}")
print(f"Min: {detections_per_frame.min()}")
print(f"Max: {detections_per_frame.max()}")