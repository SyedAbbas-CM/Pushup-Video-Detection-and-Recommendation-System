import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

# not used for now; it could be used to enchance model and do 3d pose estimation
# This is specifically used to normalize the Z (DEPTH) values of the armature points; we will need the maximum Z value
def find_max_z(folder_path):
    max_z = 0
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path) as json_file:
                data = json.load(json_file)
                for annotation in data['annotations']:
                    armature_keypoints = annotation.get('armature_keypoints', {})
                    for key, value in armature_keypoints.items():
                        z = value['z']
                        if z > max_z:
                            max_z = z
    return max_z

# convert the dictionary into a list so armature keypoints and keypoints can be normalized together
def armature_keypoints_to_list(armature_keypoints):
    keypoints_dict = {}
    for key, value in armature_keypoints.items():
        keypoints_dict[key] = [value['x'], value['y'], value['v']]
    return keypoints_dict


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

    print(f"Processing {len(json_files)} JSON files in folder {folder_path}...")

    for file in json_files:
        file_path = os.path.join(folder_path, file)
        print(f"Processing file: ________{file}_________\n\n")

        with open(file_path, 'r') as f:
            data = json.load(f)

        if "annotations" not in data:
            print(f"Warning: 'annotations' key not found in file {file}.")
            continue

        if "images" not in data:
            print(f"Warning: 'images' key not found in file {file}.")
            continue

        for annotation in data['annotations']:
            #print(annotation)
            try:
                keypoints = annotation['keypoints']
            except KeyError:
                print(f"Warning: 'keypoints' key not found in an annotation in file {file}.")
                print(f"Problematic annotation: {annotation}")
                continue

            num_keypoints = len(keypoints) // 3
            if num_keypoints < 15:
                print(f"Warning: Insufficient number of keypoints in an annotation in file {file}.")
                print(f"Problematic annotation: {annotation}")
                continue

            width = data['images'][0]['width']
            height = data['images'][0]['height']
            normalized_keypoints = normalize_keypoints(keypoints, width, height)
            all_data.append(normalized_keypoints)
            all_labels.append(label)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    return all_data, all_labels

# Example usage


folder_path = 'D:\\AIPROJECT\\preprocessed_data\\json_pushup'
folder_path2 = 'D:\\AIPROJECT\\preprocessed_data\\json_curl'
folder_path3 = 'D:\\AIPROJECT\\preprocessed_data\\json_fly'
folder_path4 = 'D:\\AIPROJECT\\preprocessed_data\\json_OHP'
folder_path5 = 'D:\\AIPROJECT\\preprocessed_data\\json_squats'
output_folder = 'D:\\AIPROJECT\\output'

# max_z = find_max_z(folder_path)
# print(max_z)

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

all_data = np.concatenate(( pushup_data,curl_data,fly_data,squat_data,  ohp_data), axis=0)
all_labels = np.concatenate((pushup_labels,curl_labels,fly_labels ,squat_labels,ohp_labels), axis=0)


# Reorder data and labels using the random indices
shuffled_data, shuffled_labels = shuffle(all_data, all_labels)


# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

# Further split the training data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

# Save it all for later use
np.save(os.path.join(output_folder, "X_train.npy"), X_train)
np.save(os.path.join(output_folder, "y_train.npy"), y_train)
np.save(os.path.join(output_folder, "X_val.npy"), X_val)
np.save(os.path.join(output_folder, "y_val.npy"), y_val)
np.save(os.path.join(output_folder, "X_test.npy"), X_test)
np.save(os.path.join(output_folder, "y_test.npy"), y_test)


#np.save(save_path, all_data)
#np.save(save_path, all_labels)
