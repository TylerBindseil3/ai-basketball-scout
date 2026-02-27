from ultralytics import YOLO
import cv2 as cv
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Initializing YOLO model
# Using v8s for higher accuracy

yolo = YOLO("yolov8s.pt")

# Create video path
video_path = BASE_DIR / 'data' / 'videos' / 'game_trimmed.mp4'

# Capture each frame for later analysis
video_capture = cv.VideoCapture(str(video_path))

# Setting up list to store coordinates in later
player_positions = []

# Initializing frame counter
frame_count = 0

# Looping through the video frame by frame
while video_capture.isOpened():
    frame_available, frame = video_capture.read()
    frame_count += 1

    # Stops the loop when we reach the end of the video
    if not frame_available:
        break

    # If there is a frame run yolo detection on it
    if frame_count % 10 == 0:
        results = yolo(frame, classes=[0], conf=.2)

        for box in results[0].boxes:
        # Get bounding box in center-x, center-y, width, height format
            x, y, w, h = box.xywh[0].cpu().numpy()
        
            # Save a dictionary for each box to list
            player_positions.append({
                'frame': frame_count,
                'x': float(x),
                'y': float(y),
                'width': float(w),
                'height': float(h)
            })

    if frame_count % 1000 == 0:
        print(f'Processed {frame_count/10} frames...')
# Close video
video_capture.release()
# Convert list into a dataframe
df = pd.DataFrame(player_positions)
# Saving data to a CSV file
output_path = BASE_DIR / 'data' / 'processed' / 'player_positions.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

# Printing outcomes
detections_per_frame = df.groupby('frame').size()
print(f"\nComplete!")
print(f"  Total detections: {len(df)}")
print(f"  Average per frame: {detections_per_frame.mean():.1f}")
print(f"  Saved to: {output_path}")