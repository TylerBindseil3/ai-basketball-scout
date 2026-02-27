import os
from dotenv import load_dotenv
import requests
import json

# Initializing Pexels url
url = "https://api.pexels.com/videos/search"

# Loading API Key from .env
load_dotenv()
pexels_key = os.getenv("PEXELS_API_KEY")

# Getting videos through API
API_KEYS = {"Authorization": pexels_key}
SEARCH_QUERY = {"query": "Basketball film"}
response = requests.get(url, headers=API_KEYS, params=SEARCH_QUERY)
data = response.json()
video = data["videos"][3]

print("Duration:", video["duration"], "seconds")
print("Number of quality options")

# Look at each quality option
for video_file in video["video_files"]:
    print(f"Quality: {video_file['quality']}, Resolution: {video_file['width']}x{video_file['height']}, Link: {video_file['link'][:50]}...")

# Downloading video
video_url = video["video_files"][1]["link"]
video_response = requests.get(video_url)

with open("game_film.mp4", "wb") as f:
    f.write(video_response.content)