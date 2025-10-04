import os
import json
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import threading
import time
from collections import deque

# Track key state for continuous jump/fall
key_state = {'w': False}

# -------------------------------
# Utilities
# -------------------------------
def max_pool(image, pool_size=(1, 2)):
    pooled_image = []
    height, width = image.shape
    pool_height, pool_width = pool_size
    for y in range(0, height, pool_height):
        for x in range(0, width, pool_width):
            region = image[y:min(y+pool_height,height), x:min(x+pool_width,width)]
            if region.size > 0:
                pooled_image.append(np.max(region))
    return np.array(pooled_image).reshape(-1)

def forward_propagation(input_vector, weights, biases):
    if input_vector.ndim > 1:
        input_vector = input_vector.flatten()
    activations = [input_vector]
    for i in range(len(weights)-1):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = np.maximum(0, z)
        activations.append(a)
    z = np.dot(activations[-1], weights[-1]) + biases[-1]
    activations.append(z)
    return activations

def reconstruct_image_from_features(features, grid_size, image_size):
    gh, gw = grid_size
    H, W = image_size
    num_grid_y = H // gh
    num_grid_x = W // gw
    reconstructable_height = num_grid_y * gh
    reconstructable_width = num_grid_x * gw

    reconstructed = np.zeros((H,W), dtype=np.float32)
    feature_size = gh*gw
    num_grids = num_grid_y * num_grid_x

    if features.size < num_grids*feature_size:
        return reconstructed

    features_reshaped = features[:num_grids*feature_size].reshape(num_grids, feature_size)

    index = 0
    for y in range(num_grid_y):
        for x in range(num_grid_x):
            grid = features_reshaped[index].reshape(gh, gw)
            reconstructed[y*gh:(y+1)*gh, x*gw:(x+1)*gw] = grid
            index += 1
    return np.clip(reconstructed, 0, 1)

def load_models(model_files):
    models = {}
    for key, model_file in model_files.items():
        with open(model_file, 'r') as f:
            model_data = json.load(f)
        weights = [np.array(w, np.float32) for w in model_data['weights']]
        biases = [np.array(b, np.float32) for b in model_data['biases']]
        config = model_data.get('config', {})
        num_input_frames = config.get('num_input_frames', 1)
        models[key] = {
            'weights': weights,
            'biases': biases,
            'num_input_frames': num_input_frames
        }
    return models

# -------------------------------
# Interactive Tkinter
# -------------------------------
def interactive_test_tk(models, start_image_path, grid_size, reshape_size):
    if not models:
        print("No models loaded!")
        return

    num_input_frames = next(iter(models.values()))['num_input_frames']

    # Load start image
    img = Image.open(start_image_path).convert('L').resize(reshape_size)
    img_np = np.array(img, dtype=np.float32)/255.0
    pooled_start = max_pool(img_np)

    # History buffer
    history_buffer = deque([pooled_start]*num_input_frames, maxlen=num_input_frames)
    current_image = img_np.copy()

    # Tkinter setup
    root = tk.Tk()
    root.title(f"Patch-based Next-frame Prediction (N={num_input_frames})")
    root.geometry("500x600")

    tk.Label(root, text=f"Sequence Length (N): {num_input_frames}").pack(pady=5)
    reconstructed_panel = tk.Label(root)
    reconstructed_panel.pack(pady=10)

    tk.Label(root, text="Creativeness (Noise)").pack()
    creativeness_scale = tk.Scale(root, from_=0.0, to=0.5, resolution=0.01, orient="horizontal")
    creativeness_scale.set(0.1)
    creativeness_scale.pack(pady=5)

    tk.Label(root, text="Press 'W' to Jump, Release for Fall ('S')").pack(pady=5)

    # -------------------------------
    def update_images(command):
        nonlocal current_image, history_buffer
        model_data = models.get(command)
        if not model_data:
            return
        weights = model_data['weights']
        biases = model_data['biases']

        # Concatenate last N pooled features
        input_vector = np.concatenate(list(history_buffer))
        acts = forward_propagation(input_vector, weights, biases)
        output_features = acts[-1]

        # Optional: apply neighbor/global weights here if available
        # output_features = output_features * consistency_weights

        # Reconstruct frame
        reconstructed_image = reconstruct_image_from_features(output_features, grid_size, reshape_size)

        # Add noise
        noise_strength = creativeness_scale.get()
        noise = np.random.uniform(-noise_strength, noise_strength, reconstructed_image.shape)
        reconstructed_image = np.clip(reconstructed_image + noise, 0, 1)

        # Display
        img_tk = ImageTk.PhotoImage(Image.fromarray((reconstructed_image*255).astype(np.uint8)).resize((450,450), Image.Resampling.NEAREST))
        reconstructed_panel.configure(image=img_tk)
        reconstructed_panel.image = img_tk

        # Update autoregressive buffer
        current_image = reconstructed_image
        new_pooled = max_pool(current_image)
        history_buffer.append(new_pooled)

    # -------------------------------
    def auto_press():
        while True:
            time.sleep(0.05)
            if key_state['w']:
                root.after(0, update_images, 'w')
            else:
                root.after(0, update_images, 's')

    def on_key_press(event):
        key = event.char.lower()
        if key == 'q':
            root.destroy()
        elif key == 'w':
            key_state['w'] = True

    def on_key_release(event):
        key = event.char.lower()
        if key == 'w':
            key_state['w'] = False

    threading.Thread(target=auto_press, daemon=True).start()
    root.bind("<KeyPress>", on_key_press)
    root.bind("<KeyRelease>", on_key_release)
    root.mainloop()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    start_image_path = "flappytest.png"  # single start image
    model_files = {
        'w': "jump_sequence.json",
        's': "fall_sequence.json"
    }
    grid_size = (2,2)
    reshape_size = (40,40)

    models = load_models(model_files)
    interactive_test_tk(models, start_image_path, grid_size, reshape_size)
