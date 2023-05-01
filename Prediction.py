import torch
import numpy as np
import json
from model2 import KeyPointsTransformer
from collections import Counter

def load_data_from_json(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)

    keypoints_list = []
    for annotation in json_data['annotations']:
        if 'keypoints' in annotation:  # Check if 'keypoints' key is present
            keypoints = annotation['keypoints']
            keypoints_list.append(keypoints)
        else:
            print(f"Skipping frame {annotation['image_id']} due to missing keypoints.")

    keypoints_array = np.array(keypoints_list)
    return keypoints_array.reshape(-1, 51)

def create_sequences(data, sequence_length):
    sequences = np.stack([data[i:i + sequence_length] for i in range(data.shape[0] - sequence_length + 1)], axis=0)
    return sequences

def predict(model, sequence):
    model.eval()

    # Convert the sequence to a tensor
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    return prediction

def simplify_output(output):
    # Extract the class predictions from the output dictionary
    class_predictions = [frame['class'] for frame in output]

    # Count the occurrences of each class
    class_counts = Counter(class_predictions)

    # Find the most common class
    most_common_class, _ = class_counts.most_common(1)[0]

    # Return the simplified output
    return most_common_class

# Load the model
model_path = "D:/AIPROJECT2/models/trained_model2.pth"
num_classes = 6
model = KeyPointsTransformer(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load frames from JSON file
json_path = "D:/AIPROJECT2/upload/json/000000.json"
frames = load_data_from_json(json_path)
print("Frames shape:", frames.shape)

# Create sequences
sequence_length = 60
sequences = create_sequences(frames, sequence_length)

# Make predictions for each sequence
predictions = []
for i, sequence in enumerate(sequences):
    prediction = predict(model, sequence)
    predictions.append({"sequence": i + 1, "class": prediction})

simplified_output = simplify_output(predictions)

print(simplified_output)
