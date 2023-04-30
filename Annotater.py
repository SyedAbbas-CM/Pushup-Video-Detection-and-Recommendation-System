
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def mediapipe_keypoints(image_path):
    # Read an image.
    image = cv2.imread(image_path)

    # Initialize MediaPipe Pose.
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        # Convert the image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run pose estimation.
        results = pose.process(image_rgb)

        # Check if pose landmarks are available.
        if not results.pose_landmarks:
            print("Pose landmarks not found.")
            return None

        # Extract keypoints.
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x * image.shape[1], landmark.y * image.shape[0], landmark.visibility])

    return keypoints



image_path = "path/to/your/image.jpg"
keypoints = mediapipe_keypoints(image_path)
if keypoints:
    print("Keypoints:", keypoints)
else:
    print("No keypoints found.")
