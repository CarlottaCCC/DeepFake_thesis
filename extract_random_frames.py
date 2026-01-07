import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm

VIDEO_DIR = "faceforensics/data/manipulated_sequences/Deepfakes/c23/videos"
OUTPUT_DIR = "faceforensics/data5/manipulated_sequences/Deepfakes/frames"
NUM_FRAMES_PER_VIDEO = 5
IMAGE_FORMAT = "jpg" 
SEED = 42

random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

print(f"Found {len(video_files)} videos")

#for each video I extract the video id and create the output directory for the frames
#there will be one folder for each video containing N frames of that video
for video_name in tqdm(video_files):
    video_path = os.path.join(VIDEO_DIR, video_name)
    video_id = Path(video_name).stem
    out_dir = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#If the number of frames of the video is more than N, I sample N frames (the indeces)
    if total_frames < NUM_FRAMES_PER_VIDEO:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = random.sample(
            range(total_frames),
            NUM_FRAMES_PER_VIDEO
        )

    frame_indices.sort()

    current_frame = 0
    saved = 0

#I go through every frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_indices:
            out_path = os.path.join(
                out_dir,
                f"frame_{current_frame:05d}.{IMAGE_FORMAT}"
            )
            cv2.imwrite(out_path, frame)
            saved += 1

        current_frame += 1
        if saved == len(frame_indices):
            break

    cap.release()