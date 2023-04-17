import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

def preprocess_data(base_dir, output_dir):
    video_dir = os.path.join(base_dir, 'dataset', 'pushups', 'pushups', 'data', 'Videos')
    json_dir = os.path.join(base_dir, 'dataset', 'pushups', 'pushups', 'data', 'Json')
    label_dir = os.path.join(base_dir, 'dataset', 'pushups', 'pushups', 'data', 'labels')

    train_output = os.path.join(output_dir, 'train')
    val_output = os.path.join(output_dir, 'val')

    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    train_videos, val_videos, _, _ = train_test_split(video_files, json_files, test_size=0.2, random_state=42)

    for video_file in train_videos:
        video_path = os.path.join(video_dir, video_file)
        json_path = os.path.join(json_dir, video_file.replace('.mp4', '.json'))
        label_path = os.path.join(label_dir, video_file.replace('.mp4', ''))

        video_output_path = os.path.join(train_output, video_file)
        json_output_path = os.path.join(train_output, video_file.replace('.mp4', '.json'))
        label_output_path = os.path.join(train_output, video_file.replace('.mp4', ''))

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0:
            print(f"Error: Video file {video_file} has unsupported format or cannot be read.")
            continue

        cap.release()

        with open(json_path, 'r') as f:
            keypoints = json.load(f)

        resampled_keypoints = []
        for idx, frame_keypoints in enumerate(keypoints):
            if idx % fps == 0:
                resampled_keypoints.append(frame_keypoints)

        with open(json_output_path, 'w') as f:
            json.dump(resampled_keypoints, f)

        shutil.copy(video_path, video_output_path)
        shutil.copytree(label_path, label_output_path)

    for video_file in val_videos:
        video_path = os.path.join(video_dir, video_file)
        json_path = os.path.join(json_dir, video_file.replace('.mp4', '.json'))
        label_path = os.path.join(label_dir, video_file.replace('.mp4', ''))

        video_output_path = os.path.join(val_output, video_file)
        json_output_path = os.path.join(val_output, video_file.replace('.mp4', '.json'))
        label_output_path = os.path.join(val_output, video_file.replace('.mp4', ''))

        shutil.copy(video_path, video_output_path)
        shutil.copy(json_path, json_output_path)
        shutil.copytree(label_path, label_output_path)



base_dir = ".."  # Assuming the dataset is one level above the project folder
output_dir = "output"  # Output folder will be created in the project folder
preprocess_data(base_dir, output_dir)
