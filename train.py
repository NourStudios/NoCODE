import os
import numpy as np
import json
from PIL import Image
import re


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


def get_brightness_values(image_path, reshape_size):
    try:
        image = Image.open(image_path).convert('L').resize(reshape_size)
        arr = np.array(image, dtype=np.float32) / 255.0
        return arr
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def max_pool(image_array, pool_size=(1, 2)):
    h, w = image_array.shape
    ph, pw = pool_size
    pooled = image_array[:h - h % ph, :w - w % pw].reshape(h // ph, ph, w // pw, pw)
    return np.max(np.max(pooled, axis=3), axis=1).flatten()


def extract_grid_features(image_array, grid_size):
    gh, gw = grid_size
    h, w = image_array.shape
    num_y, num_x = h // gh, w // gw
    features = image_array[:num_y*gh, :num_x*gw].reshape(num_y, gh, num_x, gw)
    features = features.transpose(0, 2, 1, 3).reshape(num_y*num_x, gh*gw)
    return features.flatten().astype(np.float32)


def initialize_weights_and_biases(input_size, hidden_sizes, output_size):
    weights, biases = [], []
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
        b = np.zeros(layer_sizes[i+1], dtype=np.float32)
        weights.append(w.astype(np.float32))
        biases.append(b)
    return weights, biases


def relu(x):
    return np.maximum(0, x)


def forward_propagation_batch(inputs, weights, biases):
    activations = [inputs]
    for i in range(len(weights) - 1):
        z = activations[-1] @ weights[i] + biases[i]
        a = relu(z)
        activations.append(a)
    z = activations[-1] @ weights[-1] + biases[-1]  # output layer linear
    activations.append(z)
    return activations


def backward_propagation_batch(activations, weights, biases, target_output, learning_rate):
    """Batch backward propagation and update weights/biases."""
    batch_size = target_output.shape[0]
    deltas = [activations[-1] - target_output]

    for i in range(len(weights)-2, -1, -1):
        delta = deltas[0] @ weights[i+1].T
        delta *= (activations[i+1] > 0)  # ReLU derivative
        deltas.insert(0, delta)

    for i in range(len(weights)):
        grad_w = activations[i].T @ deltas[i] / batch_size
        grad_b = np.mean(deltas[i], axis=0)
        weights[i] -= learning_rate * grad_w
        biases[i] -= learning_rate * grad_b


def train_model(data_folder, model_file, reshape_size, grid_sizes, hidden_sizes, learning_rate, iterations, num_input_frames):
    """
    Trains the model using a sequence of num_input_frames to predict a subsequent frame.
    The targets are restricted to even-indexed frames (2, 4, 6, 8, ...).
    """
    print(f"Training model using data from folder: {data_folder}")
    print(f"Using {num_input_frames} preceding frames as input.")
    print("Targets are restricted to even-indexed frames (2, 4, 6, 8, ...).")

    file_list = sorted(os.listdir(data_folder), key=natural_sort_key)
    inputs, targets = [], []
    processed_features = [] # To store the features of each frame

    # 1. Pre-process and extract features for all valid frames
    valid_file_list = []
    for file_name in file_list:
        file_path = os.path.join(data_folder, file_name)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            brightness = get_brightness_values(file_path, reshape_size)
            if brightness is not None:
                # Store both pooled (input) and gridded (target) features for every frame
                pooled = max_pool(brightness) 
                gridded = extract_grid_features(brightness, grid_sizes)
                
                processed_features.append({
                    'pooled': pooled,
                    'gridded': gridded
                })
                valid_file_list.append(file_name)

    num_valid_frames = len(processed_features)
    min_required_frames = num_input_frames + 1
    
    print(f"\nProcessed features for {num_valid_frames} valid images.")
    if num_valid_frames < min_required_frames:
        print(f"Not enough valid image frames for training. Need at least {min_required_frames}. Aborting.")
        return

    # 2. Create input/target sequences with even-index target restriction
    
    # We loop through the entire list, but ONLY use the even indices (1, 3, 5, 7, ...) as the target frame index.
    # The actual frame indices (0-based) for the targets are 1, 3, 5, 7, ...
    # This corresponds to file names 2.png, 4.png, 6.png, 8.png, ...

    # The loop iterates on the target frame index (i)
    # Start at the first even index (1) and step by 2
    for i in range(1, num_valid_frames, 2):
        # 1. Define the Target (Target frame is at index i, which is 1, 3, 5, ...)
        # The target feature is the gridded feature of the even-indexed frame.
        target_frame_index = i
        targets.append(processed_features[target_frame_index]['gridded'])
        
        # 2. Define the Input Sequence (N frames preceding the target frame)
        # The sequence starts at index: i - num_input_frames
        # The sequence ends at index: i - 1
        sequence_start_index = i - num_input_frames
        sequence_end_index = i # The slice will go up to but not include this index

        # If the start of the sequence goes before the first frame (index 0), we must skip this sample
        if sequence_start_index < 0:
            print(f"Skipping target frame at index {i} ('{valid_file_list[i]}') because the required sequence of {num_input_frames} frames is not available.")
            targets.pop() # Remove the target we just appended
            continue
        
        # Concatenate the 'pooled' features for the input sequence
        input_sequence = [processed_features[j]['pooled'] 
                          for j in range(sequence_start_index, sequence_end_index)]
            
        inputs.append(np.concatenate(input_sequence))


    num_pairs = len(inputs)
    print(f"Created {num_pairs} training pairs.")
    if num_pairs == 0:
        print("No training pairs created. Aborting.")
        return

    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    # The input size is now N * (size of a single frame's pooled feature)
    input_size = inputs.shape[1] 
    output_size = targets.shape[1]
    
    # Check if input size is consistent with num_input_frames
    expected_single_pooled_size = processed_features[0]['pooled'].size
    if input_size != num_input_frames * expected_single_pooled_size:
        raise ValueError("Calculated input size mismatch. Check feature extraction logic.")

    weights, biases = initialize_weights_and_biases(input_size, hidden_sizes, output_size)

    # 3. Training Loop and Model Saving (remains the same)
    for iteration in range(iterations):
        activations = forward_propagation_batch(inputs, weights, biases)
        backward_propagation_batch(activations, weights, biases, targets, learning_rate)

        if iteration % 1000 == 0 or iteration == iterations - 1:
            loss = np.mean((activations[-1] - targets) ** 2)
            print(f"Iteration {iteration}, Loss: {loss:.4f}")

    model_data = {'weights': [w.tolist() for w in weights],
                  'biases': [b.tolist() for b in biases],
                  'config': {'num_input_frames': num_input_frames,
                             'reshape_size': reshape_size,
                             'grid_sizes': grid_sizes}}
    with open(model_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"Model trained and saved to {model_file}")


if __name__ == "__main__":
    data_folder = "fall"
    model_file = "fall_sequence_odd_target.json"
    reshape_size = (28, 28)
    grid_sizes = [2, 2]
    hidden_sizes = [1000, 1000]
    learning_rate = 0.001
    iterations = 10000
    num_input_frames = 3 

    train_model(data_folder, model_file, reshape_size, grid_sizes, hidden_sizes, 

                learning_rate, iterations, num_input_frames=num_input_frames)

