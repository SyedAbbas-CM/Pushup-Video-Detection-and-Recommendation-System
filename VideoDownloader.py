import os
import subprocess
import json

def download_youtube_video(url, output_folder):
    try:
        command = [
            'yt-dlp',
            '--format', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            '--output', os.path.join(output_folder, '%(title)s.%(ext)s'),
            url
        ]
        subprocess.run(command, check=True)
        print(f"Downloaded: {url}")
    except Exception as e:
        print(f"Error: Failed to download video {url}. Error: {e}")

def download_kinetics_videos(json_data, output_folder, subset='train', num_videos=None):
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for video_id, video_info in json_data.items():
        if video_info['subset'] == subset:
            url = video_info['url']
            print(f"Downloading video {count + 1}: {url}")
            download_youtube_video(url, output_folder)

            count += 1
            if num_videos and count >= num_videos:
                break

# Load the Kinetics JSON data.
with open("D:\\AIPROJECT2\\preprocessed_data\\YoutubeVideosLinksJson\\train.json", "r") as f:
    kinetics_data = json.load(f)

# Download videos.
output_folder = "D:\\AIPROJECT2\\preprocessed_data\\NonExcerciseVideos"
download_kinetics_videos(kinetics_data, output_folder, subset='train', num_videos=250)  # Download 10 train videos
