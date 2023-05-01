import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

label_to_index = {"pushup": 0, "curl": 1, "fly": 2, "squat": 3, "ohp": 4, "non_exercise": 5}

keypoint_names = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]
dtype = []
for name in keypoint_names:
    dtype.append((name, np.float32, (3,)))

def normalize_keypoints(keypoints, width, height):
    normalized_keypoints = []
    for i in range(0, len(keypoints), 3):
        x = keypoints[i] / width
        y = keypoints[i + 1] / height
        v = keypoints[i + 2]
        normalized_keypoints.extend([x, y, v])
    #print("normalization function returns:")
    #print(normalized_keypoints)
    return normalized_keypoints

def create_sequences(data, sequence_length, step_size):
    sequences = np.stack([data[i:i + sequence_length] for i in range(0, data.shape[0] - sequence_length + 1, step_size)], axis=0)
    return sequences

sequence_length = 60  # Adjust this based on your requirements
step_size = 10  # Adjust this based on your requirements


def preprocess_data(folder_path, label):
    all_data = []
    all_labels = []

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    if not json_files:
        print(f"Error: No JSON files found in folder {folder_path}.")
        return

    for file in json_files:
        file_data = []  # Store the data for the current file
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'r') as f:
            data = json.load(f)

        if "annotations" not in data:
            continue

        if "images" not in data:
            width = 224
            height = 224
        else:
            width = data['images'][0]['width']
            height = data['images'][0]['height']

        for annotation in data['annotations']:
            try:
                keypoints = annotation['keypoints']
            except KeyError:
                continue

            num_keypoints = len(keypoints) // 3
            if num_keypoints < 15:
                continue

            normalized_keypoints = normalize_keypoints(keypoints, width, height)
            file_data.append(normalized_keypoints)

        # Create sequences from the file_data if there are enough frames
        if len(file_data) >= sequence_length:
            file_data = np.array(file_data)
            file_sequences = create_sequences(file_data, sequence_length, step_size)
            file_labels = np.full(file_sequences.shape[0], label_to_index[label])

            all_data.extend(file_sequences)
            all_labels.extend(file_labels)

        print(f"Appended label {label} for file {file}")

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    print(f"Data length for {label}: {len(all_data)}")
    print(f"Labels length for {label}: {len(all_labels)}")
    return all_data, all_labels

# Example usage

folder_path = 'D:\\AIPROJECT2\\preprocessed_data\\json_pushup'
folder_path2 = 'D:\\AIPROJECT2\\preprocessed_data\\json_curl'
folder_path3 = 'D:\\AIPROJECT2\\preprocessed_data\\json_fly'
folder_path4 = 'D:\\AIPROJECT2\\preprocessed_data\\json_OHP'
folder_path5 = 'D:\\AIPROJECT2\\preprocessed_data\\json_squats'
folder_path6 = 'D:\\AIPROJECT2\\preprocessed_data\\json_NonExcercise'
output_folder = 'D:\\AIPROJECT2\\output'

NoExcercise, NoExcercise_labels = preprocess_data(folder_path6, "non_exercise")
print("No Ecercise data preprocessing complated!")

pushup_data, pushup_labels = preprocess_data(folder_path, "pushup")
print("Pushup data preprocessing complated!")
curl_data, curl_labels = preprocess_data(folder_path2, "curl")
print("Curl data preprocessing complated!")
fly_data, fly_labels = preprocess_data(folder_path3, "fly")
print("FLY data preprocessing complated!")
squat_data, squat_labels = preprocess_data(folder_path4, "squat")
print("SQUAT data preprocessing complated!")
ohp_data, ohp_labels = preprocess_data(folder_path5, "ohp")
print("OHP data preprocessing complated!")

print(pushup_data.shape)
print(curl_data.shape)
print(fly_data.shape)
print(squat_data.shape)
print(ohp_data.shape)
print(NoExcercise.shape)


all_data = np.concatenate(( pushup_data,curl_data,fly_data,squat_data,  ohp_data,NoExcercise), axis=0)
all_labels = np.concatenate((pushup_labels,curl_labels,fly_labels ,squat_labels,ohp_labels,NoExcercise_labels), axis=0)


# Reorder data and labels using the random indices
shuffled_data, shuffled_labels = shuffle(all_data, all_labels)


# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

# Further split the training data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)


np.save(os.path.join(output_folder, "X_train.npy"), X_train)
np.save(os.path.join(output_folder, "X_test.npy"), X_test)
np.save(os.path.join(output_folder, "X_val.npy"), X_val)
np.save(os.path.join(output_folder, "y_train.npy"), y_train)
np.save(os.path.join(output_folder, "y_test.npy"), y_test)
np.save(os.path.join(output_folder, "y_val.npy"), y_val)
