import os
import numpy as np
import json
from PIL import Image
import re


def natural_sort_key(s):
    """
    Extracts integers from a string for natural sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


def get_brightness_values(image_path, reshape_size):
    """Loads an image, converts it to grayscale, and resizes it."""
    try:
        image = Image.open(image_path)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize(reshape_size)  # Resize to given size
        brightness_values = np.array(image, dtype=np.float32)  # Use float32 for efficiency
        brightness_values = brightness_values / 255.0  # Normalize to [0, 1]
        return brightness_values
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def max_pool(image_array, pool_size=(2, 2)):
    """Performs max pooling on the image using the specified pool size."""
    pooled_image = []
    height, width = image_array.shape
    pool_height, pool_width = pool_size

    for y in range(0, height, pool_height):
        for x in range(0, width, pool_width):
            pool_region = image_array[y:y + pool_height, x:x + pool_width]
            pooled_image.append(np.max(pool_region))  # Max pooling in the region

    return np.array(pooled_image).reshape(-1)  # Flatten to 1D


def extract_grid_features(image_array, grid_size):
    """Extracts grid features from the image."""
    features = []
    height, width = image_array.shape
    grid_height, grid_width = grid_size

    num_grid_y = height // grid_height
    num_grid_x = width // grid_width

    for y in range(num_grid_y):
        for x in range(num_grid_x):
            grid = image_array[y * grid_height:(y + 1) * grid_height,
                               x * grid_width:(x + 1) * grid_width]
            features.append(grid.flatten())  # Save raw grid data as a feature vector

    return np.array(features, dtype=np.float32)  # Use float32 for efficiency


def initialize_weights_and_biases(input_size, hidden_sizes, output_size):
    """Initialize weights and biases for the neural network."""
    weights = []
    biases = []

    # Initialize weights and biases for each layer
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
        b = np.zeros(layer_sizes[i+1])
        weights.append(w)
        biases.append(b)

    return weights, biases


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def forward_propagation(input_vector, weights, biases):
    """Forward propagation through the neural network."""
    activations = [input_vector]

    for i in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = relu(z)
        activations.append(a)

    # Output layer is linear (no activation)
    z = np.dot(activations[-1], weights[-1]) + biases[-1]
    activations.append(z)

    return activations


def backward_propagation(activations, weights, biases, target_output, learning_rate):
    """Backward propagation and weight update."""
    deltas = []
    output_error = activations[-1] - target_output  # Error at output layer
    deltas.append(output_error)

    # Backpropagate the error
    for i in range(len(weights) - 2, -1, -1):
        delta = np.dot(deltas[-1], weights[i+1].T) * (activations[i+1] > 0)
        deltas.append(delta)

    deltas.reverse()

    # Update weights and biases
    for i in range(len(weights)):
        weights[i] -= learning_rate * np.outer(activations[i], deltas[i])
        biases[i] -= learning_rate * deltas[i]


def calculate_loss(inputs, targets, weights, biases):
    """Calculates Mean Squared Error loss."""
    total_loss = 0
    for input_vector, target_output in zip(inputs, targets):
        activations = forward_propagation(input_vector, weights, biases)
        loss = np.mean((activations[-1] - target_output) ** 2)
        total_loss += loss
    return total_loss / len(inputs)


def train_model(data_folder, model_file, reshape_size, grid_sizes, hidden_sizes, learning_rate, iterations):
    """Train the model with the specified parameters."""
    print(f"Training model using data from folder: {data_folder}")

    # Prepare inputs (max-pooled images) and targets (next image features)
    inputs = []
    targets = []

    # Use natural sorting for filenames
    file_list = sorted(os.listdir(data_folder), key=natural_sort_key)

    skipped_files = []  # To track skipped files
    processed_files = []  # To track successfully processed files

    for i in range(0, len(file_list) - 1, 2):  # Skip every second image (i.e., pair 1 with 2, 3 with 4, etc.)
        current_file_path = os.path.join(data_folder, file_list[i])
        next_file_path = os.path.join(data_folder, file_list[i + 1])

        # Validate file extensions
        if not current_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping unsupported file: {current_file_path}")
            skipped_files.append(current_file_path)
            continue
        if not next_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping unsupported file: {next_file_path}")
            skipped_files.append(next_file_path)
            continue

        # Process current image
        current_brightness = get_brightness_values(current_file_path, reshape_size)
        if current_brightness is None:
            print(f"Skipping unreadable image: {current_file_path}")
            skipped_files.append(current_file_path)
            continue

        # Process next image (target output)
        next_brightness = get_brightness_values(next_file_path, reshape_size)
        if next_brightness is None:
            print(f"Skipping unreadable image: {next_file_path}")
            skipped_files.append(next_file_path)
            continue

        # Perform max pooling on the current image (input)
        pooled_image = max_pool(current_brightness)

        # Extract features from the next image (target)
        next_features = extract_grid_features(next_brightness, grid_sizes)

        inputs.append(pooled_image)  # Use max-pooled image as input
        targets.append(next_features.flatten())  # Flatten grid features of next image as target

        # Log successfully processed files
        processed_files.append((current_file_path, next_file_path))

    # Log summary of skipped files
    print(f"\nProcessed {len(processed_files)} file pairs.")
    print(f"Skipped {len(skipped_files)} files due to errors or unsupported formats.")
    if skipped_files:
        print("Skipped files:")
        for file in skipped_files:
            print(f"  - {file}")

    # Proceed with training if we have enough data
    if len(inputs) < 2:  # Need at least two images for meaningful training
        print("Not enough valid image pairs for training. Aborting.")
        return

    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    # Initialize weights and biases
    input_size = inputs.shape[1]
    output_size = targets.shape[1]
    weights, biases = initialize_weights_and_biases(input_size, hidden_sizes, output_size)

    # Training loop
    for iteration in range(iterations):
        for input_vector, target_output in zip(inputs, targets):
            activations = forward_propagation(input_vector, weights, biases)
            backward_propagation(activations, weights, biases, target_output, learning_rate)

        if iteration % 10 == 0:
            loss = calculate_loss(inputs, targets, weights, biases)
            print(f"Iteration {iteration}, Loss: {loss:.4f}")

    # Save model
    model_data = {
        'weights': [w.tolist() for w in weights],
        'biases': [b.tolist() for b in biases],
    }

    with open(model_file, 'w') as output_file:
        json.dump(model_data, output_file, indent=2)

    print(f"Model trained and saved to {model_file}.")

if __name__ == "__main__":
    data_folder = "fall"
    model_file = "fall.json"  # Output model file
    reshape_size = (28, 28)  # Resize images to 28x28
    grid_sizes = [2, 2]  # Grid size
    hidden_sizes = []  # Example hidden layer sizes
    learning_rate = 0.001  # Learning rate
    iterations = 300  # Number of iterations

    train_model(data_folder, model_file, reshape_size, grid_sizes, hidden_sizes, learning_rate, iterations)
