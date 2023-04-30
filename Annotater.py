import cv2
import mediapipe as mp
import os
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def mediapipe_keypoints(image_path):
    relevant_keypoints = [0, 3, 6, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    image = cv2.imread(image_path)

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("Pose landmarks not found.")
            return None

        keypoints = []
        for i in relevant_keypoints:
            landmark = results.pose_landmarks.landmark[i]
            keypoints.extend([int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]), int(landmark.visibility * 2)])

    return keypoints

output_base_folder = "D:\\AIPROJECT2\\preprocessed_data\\ProcessedVideos"
output_json_folder = "D:\\AIPROJECT2\\preprocessed_data\\json_NonExcercise"

if not os.path.exists(output_json_folder):
    os.makedirs(output_json_folder)

json_file_index = 0

for video_folder in os.listdir(output_base_folder):
    video_output_folder = os.path.join(output_base_folder, video_folder)
    annotations = []

    for frame_file in os.listdir(video_output_folder):
        frame_path = os.path.join(video_output_folder, frame_file)

        keypoints = mediapipe_keypoints(frame_path)
        if keypoints:
            annotation = {"id": len(annotations), "keypoints": keypoints}
            annotations.append(annotation)
        else:
            print(f"No keypoints found for {frame_file}.")

    json_filename = f"{json_file_index:06d}.json"
    json_file_path = os.path.join(output_json_folder, json_filename)

    with open(json_file_path, "w") as json_file:
        json.dump({"annotations": annotations}, json_file, indent=4,separators=(',', ': '))

    json_file_index += 1
    print(f"JSON file saved: {json_file_path}")

