import os
import numpy as np
import json
from PIL import Image, ImageTk
import tkinter as tk
import threading
import time

# Global variable to track the last detected key
last_key = None

def max_pool(image, pool_size=(2, 2)):
    """Performs max pooling on the image using the specified pool size."""
    pooled_image = []
    height, width = image.shape
    pool_height, pool_width = pool_size

    for y in range(0, height, pool_height):
        for x in range(0, width, pool_width):
            pool_region = image[y:y + pool_height, x:x + pool_width]
            pooled_image.append(np.max(pool_region))  # Max pooling in the region

    return np.array(pooled_image).reshape(-1)  # Flatten to 1D

def forward_propagation(input_vector, weights, biases):
    """Forward propagation through the neural network."""
    activations = [input_vector]
    for i in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = np.maximum(0, z)  # ReLU activation
        activations.append(a)
    # Output layer
    z = np.dot(activations[-1], weights[-1]) + biases[-1]
    activations.append(z)  # No activation (raw output)
    return activations

def reconstruct_image_from_features(features, grid_size, image_size):
    """Reconstructs an image from 2x2 grids (features)."""
    grid_height, grid_width = grid_size
    height, width = image_size

    reconstructed_image = np.zeros((height, width))
    num_grid_y = height // grid_height
    num_grid_x = width // grid_width

    index = 0
    for y in range(num_grid_y):
        for x in range(num_grid_x):
            # Reshape feature into 2x2 grid
            grid = features[index].reshape(grid_size)
            # Place the grid into the corresponding position in the image
            reconstructed_image[y * grid_height:(y + 1) * grid_height,
                                x * grid_width:(x + 1) * grid_width] = grid
            index += 1

    return np.clip(reconstructed_image, 0, 1)

def load_models(model_files):
    """Load all models from the given dictionary of JSON files."""
    models = {}
    for key, model_file in model_files.items():
        with open(model_file, 'r') as file:
            model_data = json.load(file)
        weights = [np.array(w) for w in model_data['weights']]
        biases = [np.array(b) for b in model_data['biases']]
        models[key] = (weights, biases)
    return models

def interactive_test_tk(models, test_image_path, output_dir, grid_size, reshape_size):
    """Interactive testing with Tkinter GUI."""
    os.makedirs(output_dir, exist_ok=True)

    # Load the initial test image
    current_image = Image.open(test_image_path).convert('L')
    current_image = current_image.resize(reshape_size)
    current_image = np.array(current_image, dtype=np.float32) / 255.0

    pooled_image = max_pool(current_image)

    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Interactive Model Test")
    root.geometry("400x200")

    # Image panel
    current_panel = tk.Label(root)
    reconstructed_panel = tk.Label(root)
    current_panel.pack(side="left", padx=10, pady=10)
    reconstructed_panel.pack(side="right", padx=10, pady=10)

    counter = 1  # Starting counter for image saving

    def update_images(command):
        nonlocal current_image, pooled_image, counter

        if command not in models:
            print("Invalid command.")
            return

        weights, biases = models[command]

        # Forward propagate the input through the selected model
        input_vector = pooled_image
        activations = forward_propagation(input_vector, weights, biases)
        output_features = activations[-1]  # Model's output layer activations

        # Reconstruct the image
        num_grids = (reshape_size[0] // grid_size[0]) * (reshape_size[1] // grid_size[1])
        feature_size = grid_size[0] * grid_size[1]
        output_features = output_features[:num_grids * feature_size].reshape((num_grids, feature_size))
        reconstructed_image = reconstruct_image_from_features(output_features, grid_size, reshape_size)

        # Convert images for display
        current_image_pil = Image.fromarray((current_image * 255).astype(np.uint8))
        reconstructed_image_pil = Image.fromarray((reconstructed_image * 255).astype(np.uint8))

        # Update Tkinter image panels
        current_img_tk = ImageTk.PhotoImage(current_image_pil.resize((400, 400)))
        reconstructed_img_tk = ImageTk.PhotoImage(reconstructed_image_pil.resize((400, 400)))

        current_panel.configure(image=current_img_tk)
        current_panel.image = current_img_tk

        reconstructed_panel.configure(image=reconstructed_img_tk)
        reconstructed_panel.image = reconstructed_img_tk

        # Save images with sequential numbering
        current_image_path = os.path.join(output_dir, f"{counter}.png")
        reconstructed_image_path = os.path.join(output_dir, f"{counter + 1}.png")

        current_image_pil.save(current_image_path)
        reconstructed_image_pil.save(reconstructed_image_path)

        # Increment the counter for the next images
        counter += 2  # Since two images are saved in each update (current and reconstructed)

        # Update current image for the next iteration
        current_image = reconstructed_image
        pooled_image = max_pool(current_image)

    def auto_press():
        global last_key
        last_w_press_time = time.time()  # Track the last time 'w' was pressed
        
        while True:
            time.sleep(0.1)  # Check every 0.1 seconds

            # Reset last_key to None every 0.1 seconds
            last_key = None

            # Check if 'w' was pressed recently
            if last_key == 'w':
                last_w_press_time = time.time()  # Update the last press time
                update_images('w')

            # If 'w' hasn't been pressed for 0.1 seconds, default to 's'
            elif time.time() - last_w_press_time > 0.1:
                update_images('s')

    def on_key_press(event):
        global last_key
        key = event.char.lower()
        if key == 'q':
            root.destroy()
        elif key in models:
            last_key = key  # Update the last detected key
            update_images(key)

    # Start auto-press thread
    auto_press_thread = threading.Thread(target=auto_press, daemon=True)
    auto_press_thread.start()

    root.bind("<KeyPress>", on_key_press)
    root.mainloop()

# Script Execution
if __name__ == "__main__":
    test_image_path = "1 copy 28.png"  # Path to a test image
    model_files = {
        'w': "fall.json",
        'a': "a.json",
        's': "jump.json",
        'd': "d.json"
    }
    output_dir = "generated"  # Output directory for saving generated images
    grid_size = (2, 2)  # Size of each feature grid (2x2)
    reshape_size = (28, 28)  # Image size used during training

    models = load_models(model_files)
    interactive_test_tk(models, test_image_path, output_dir, grid_size, reshape_size)
