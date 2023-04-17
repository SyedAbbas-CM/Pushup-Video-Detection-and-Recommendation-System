import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '1'
import cv2
import json
import shutil
from distutils import dir_util

def preprocess_data(base_dir, output_dir):
    video_dir = os.path.join(base_dir, "dataset", "pushups", "pushups", "data", "videos")
    json_dir = os.path.join(base_dir, "dataset", "pushups", "pushups", "data", "json")
    label_dir = os.path.join(base_dir, "dataset", "pushups", "pushups", "data", "labels")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)
        json_path = os.path.join(json_dir, os.path.splitext(video_name)[0] + ".json")
        label_path = os.path.join(label_dir, os.path.splitext(video_name)[0] + "_img_labels")

        if not os.path.exists(video_path) or not os.path.exists(json_path):
            print(f"Error: Video file {video_name} or JSON file {os.path.splitext(video_name)[0]}.json not found.")
            continue

        video_output_dir = os.path.join(output_dir, "videos", os.path.splitext(video_name)[0])
        json_output_path = os.path.join(output_dir, "json", os.path.splitext(video_name)[0] + ".json")
        label_output_path = os.path.join(output_dir, "labels", os.path.splitext(video_name)[0])

        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        if not os.path.exists(os.path.dirname(json_output_path)):
            os.makedirs(os.path.dirname(json_output_path))
        if not os.path.exists(label_output_path) and os.path.exists(label_path):
            os.makedirs(label_output_path)

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            if idx % fps == 0:
                frame_output_path = os.path.join(video_output_dir, f"{idx:05d}.png")
                cv2.imwrite(frame_output_path, frame)

        cap.release()

        shutil.copy(json_path, json_output_path)
        if os.path.exists(label_path):
            dir_util.copy_tree(label_path, label_output_path)

base_dir = ".."
output_dir = "preprocessed_data"
preprocess_data(base_dir, output_dir)
