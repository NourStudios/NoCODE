import os
import numpy as np
import json
from PIL import Image, ImageTk
import tkinter as tk
import threading
import time

# Track key state for jump/fall
key_state = {'w': False}

def max_pool(image, pool_size=(1, 2)):
    pooled_image = []
    height, width = image.shape
    pool_height, pool_width = pool_size

    for y in range(0, height, pool_height):
        for x in range(0, width, pool_width):
            pool_region = image[y:y + pool_height, x:x + pool_width]
            pooled_image.append(np.max(pool_region))
    return np.array(pooled_image).reshape(-1)

def forward_propagation(input_vector, weights, biases):
    activations = [input_vector]
    for i in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = np.maximum(0, z)
        activations.append(a)
    z = np.dot(activations[-1], weights[-1]) + biases[-1]
    activations.append(z)
    return activations

def reconstruct_image_from_features(features, grid_size, image_size):
    grid_height, grid_width = grid_size
    height, width = image_size
    reconstructed_image = np.zeros((height, width))
    num_grid_y = height // grid_height
    num_grid_x = width // grid_width

    index = 0
    for y in range(num_grid_y):
        for x in range(num_grid_x):
            grid = features[index].reshape(grid_size)
            reconstructed_image[y * grid_height:(y + 1) * grid_height,
                                x * grid_width:(x + 1) * grid_width] = grid
            index += 1
    return np.clip(reconstructed_image, 0, 1)

def load_models(model_files):
    models = {}
    for key, model_file in model_files.items():
        with open(model_file, 'r') as file:
            model_data = json.load(file)
        weights = [np.array(w) for w in model_data['weights']]
        biases = [np.array(b) for b in model_data['biases']]
        models[key] = (weights, biases)
    return models

def interactive_test_tk(models, test_image_path, grid_size, reshape_size):
    # Load initial test image
    current_image = Image.open(test_image_path).convert('L')
    current_image = current_image.resize(reshape_size)
    current_image = np.array(current_image, dtype=np.float32) / 255.0
    pooled_image = max_pool(current_image)

    # Tkinter setup
    root = tk.Tk()
    root.title("Interactive Model Test")
    root.geometry("500x650")

    # Reconstructed frame display
    reconstructed_panel = tk.Label(root)
    reconstructed_panel.pack(pady=10)

    # Creativeness slider
    tk.Label(root, text="Creativeness (Noise)").pack()
    creativeness_scale = tk.Scale(root, from_=0.0, to=0.5, resolution=0.01, orient="horizontal", length=400)
    creativeness_scale.set(0.1)
    creativeness_scale.pack(pady=10)

    # Update image function
    def update_images(command):
        nonlocal current_image, pooled_image
        if command not in models:
            return
        weights, biases = models[command]
        activations = forward_propagation(pooled_image, weights, biases)
        output_features = activations[-1]

        num_grids = (reshape_size[0] // grid_size[0]) * (reshape_size[1] // grid_size[1])
        feature_size = grid_size[0] * grid_size[1]
        output_features = output_features[:num_grids * feature_size].reshape((num_grids, feature_size))
        reconstructed_image = reconstruct_image_from_features(output_features, grid_size, reshape_size)

        # Add adjustable noise
        noise_strength = creativeness_scale.get()
        noise = np.random.uniform(-noise_strength, noise_strength, reconstructed_image.shape).astype(np.float32)
        reconstructed_image = np.clip(reconstructed_image + noise, 0, 1)

        # Display image
        reconstructed_image_pil = Image.fromarray((reconstructed_image * 255).astype(np.uint8))
        reconstructed_img_tk = ImageTk.PhotoImage(reconstructed_image_pil.resize((450, 450)))
        reconstructed_panel.configure(image=reconstructed_img_tk)
        reconstructed_panel.image = reconstructed_img_tk

        # Update for next iteration
        current_image = reconstructed_image
        pooled_image = max_pool(current_image)

    # Auto update loop
    def auto_press():
        while True:
            time.sleep(0.05)
            if key_state['w']:
                update_images('w')  # Jump
            else:
                update_images('s')  # Fall

    # Mobile full-screen touch events
    def on_screen_touch(event):
        key_state['w'] = True

    def on_screen_release(event):
        key_state['w'] = False

    # Bindings
    root.bind("<ButtonPress-1>", on_screen_touch)
    root.bind("<ButtonRelease-1>", on_screen_release)

    # Start auto-update thread
    threading.Thread(target=auto_press, daemon=True).start()
    root.mainloop()

# Script execution
if __name__ == "__main__":
    test_image_path = "flappytest.png"
    model_files = {
        'w': "jump.json",
        's': "fall.json",
    }
    grid_size = (2, 2)
    reshape_size = (28, 28)

    models = load_models(model_files)
    interactive_test_tk(models, test_image_path, grid_size, reshape_size)
