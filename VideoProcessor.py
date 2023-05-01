import cv2
import os

def process_video(input_video_path, output_folder, max_frames=1000, target_size=(224, 224)):
    print(f"Processing video: {input_video_path}")
    # Open the video file.
    video = cv2.VideoCapture(input_video_path)

    if not video.isOpened():
        print(f"Error: Unable to open video file: {input_video_path}")
        return

    # Get the total number of frames in the video.
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame skipping rate based on the total number of frames and the desired max number of frames.
    frame_skip_rate = max(int(total_frames / max_frames), 1)
    print(f"Total frames: {total_frames}, frame skip rate: {frame_skip_rate}")

    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    output_frame_count = 0
    while True:
        # Read a frame from the video.
        ret, frame = video.read()

        # Break the loop if we have reached the end of the video.
        if not ret:
            break

        # Process every nth frame, where n is the frame_skip_rate.
        if frame_count % frame_skip_rate == 0:
            # Resize the frame.
            resized_frame = cv2.resize(frame, target_size)

            # Save the resized frame to the output folder.
            output_frame_path = os.path.join(output_folder, f"frame{output_frame_count}.jpg")
            cv2.imwrite(output_frame_path, resized_frame)

            output_frame_count += 1

        frame_count += 1

    print(f"Processed frames: {output_frame_count}")

    # Release the video file.
    video.release()

videos_folder = "D:\\AIPROJECT2\\upload"
output_base_folder = "D:\\AIPROJECT2\\upload\\uploaded_frames"
max_frames = 350

for video_file in os.listdir(videos_folder):
    input_video_path = os.path.join(videos_folder, video_file)
    output_folder = os.path.join(output_base_folder, os.path.splitext(video_file)[0])
    process_video(input_video_path, output_folder, max_frames)
