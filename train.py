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


def train_model(data_folder, model_file, reshape_size, grid_sizes, hidden_sizes, learning_rate, iterations):
    print(f"Training model using data from folder: {data_folder}")

    file_list = sorted(os.listdir(data_folder), key=natural_sort_key)
    inputs, targets = [], []

    skipped_files, processed_files = [], []

    for i in range(0, len(file_list) - 1, 2):
        current_file_path = os.path.join(data_folder, file_list[i])
        next_file_path = os.path.join(data_folder, file_list[i+1])

        if not current_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            skipped_files.append(current_file_path)
            continue
        if not next_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            skipped_files.append(next_file_path)
            continue

        current_brightness = get_brightness_values(current_file_path, reshape_size)
        next_brightness = get_brightness_values(next_file_path, reshape_size)
        if current_brightness is None or next_brightness is None:
            skipped_files.extend([current_file_path, next_file_path])
            continue

        pooled_image = max_pool(current_brightness)
        next_features = extract_grid_features(next_brightness, grid_sizes)

        inputs.append(pooled_image)
        targets.append(next_features)
        processed_files.append((current_file_path, next_file_path))

    print(f"\nProcessed {len(processed_files)} file pairs.")
    print(f"Skipped {len(skipped_files)} files.")

    if len(inputs) < 2:
        print("Not enough valid image pairs for training. Aborting.")
        return

    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    input_size = inputs.shape[1]
    output_size = targets.shape[1]
    weights, biases = initialize_weights_and_biases(input_size, hidden_sizes, output_size)

    for iteration in range(iterations):
        activations = forward_propagation_batch(inputs, weights, biases)
        backward_propagation_batch(activations, weights, biases, targets, learning_rate)

        if iteration % 10 == 0:
            loss = np.mean((activations[-1] - targets) ** 2)
            print(f"Iteration {iteration}, Loss: {loss:.4f}")

    model_data = {'weights': [w.tolist() for w in weights],
                  'biases': [b.tolist() for b in biases]}
    with open(model_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"Model trained and saved to {model_file}.")


if __name__ == "__main__":
    data_folder = "fall"
    model_file = "fall.json"
    reshape_size = (28, 28)
    grid_sizes = [2, 2]
    hidden_sizes = [1000, 1000] 
    learning_rate = 0.001
    iterations = 10000

    train_model(data_folder, model_file, reshape_size, grid_sizes, hidden_sizes, learning_rate, iterations)

