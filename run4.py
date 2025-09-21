import os
import json
import time
import threading
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
import re
from scipy.spatial import distance

# ---------------------------
# Utilities
# ---------------------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_and_preprocess_rgb(image_path, reshape_size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(reshape_size)
    return np.array(image, dtype=np.float32) / 255.0

def split_into_patches(image_array, patch_size, flatten=True):
    h, w, c = image_array.shape
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image_array[y:y+patch_size, x:x+patch_size, :]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                if flatten:
                    patches.append(patch.flatten())
                else:
                    patches.append(patch)
    return np.array(patches)

def decode_frame(ids, patch_size, grid_size, codebook_centroids):
    h, w = grid_size
    frame = np.zeros((h*patch_size, w*patch_size, 3))
    idx = 0
    for y in range(h):
        for x in range(w):
            patch = codebook_centroids[ids[idx]].reshape(patch_size, patch_size, 3)
            frame[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size, :] = patch
            idx += 1
    return frame

def get_brightness_values(image_array):
    if image_array.ndim == 2:
        return image_array
    return np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])

def max_pool(image_array, pool_size=(2, 2)):
    pooled_image = []
    height, width = image_array.shape
    pool_height, pool_width = pool_size
    for y in range(0, height, pool_height):
        for x in range(0, width, pool_width):
            pool_region = image_array[y:y + pool_height, x:x + pool_width]
            pooled_image.append(np.max(pool_region))
    return np.array(pooled_image).reshape(-1)

def relu(x):
    return np.maximum(0, x)

def forward_propagation(input_vector, weights, biases):
    activations = [input_vector]
    for i in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = relu(z)
        activations.append(a)
    z = np.dot(activations[-1], weights[-1]) + biases[-1]
    activations.append(z)
    return activations

def forward_patch_model(x, params):
    W1, b1, W2, b2 = params
    h = np.maximum(0, np.dot(x, W1) + b1)
    logits = np.dot(h, W2) + b2
    return logits

class DummyKMeans:
    def __init__(self, cluster_centers):
        self.cluster_centers_ = cluster_centers
        self.n_clusters = cluster_centers.shape[0]
    def predict(self, X):
        distances_to_centroids = distance.cdist(X, self.cluster_centers_, 'euclidean')
        return np.argmin(distances_to_centroids, axis=1)

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

# ---------------------------
# Load patch dataset for denoising
# ---------------------------
patch_data = np.load("patch_dataset/patch_dataset.npz")
noisy_library = patch_data["noisy"]   # shape (N, 8, 8, 3)
clean_library = patch_data["clean"]   # shape (N, 8, 8, 3)

def find_closest_patch(patch, noisy_library, clean_library):
    patch_flat = patch.flatten()
    diffs = noisy_library.reshape(len(noisy_library), -1) - patch_flat
    dists = np.sum(diffs**2, axis=1)
    idx = np.argmin(dists)
    return clean_library[idx]

def pad_to_multiple(image, multiple=8):
    h, w, c = image.shape
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    padded = np.zeros((new_h, new_w, c), dtype=image.dtype)
    padded[:h, :w, :] = image
    return padded, h, w

def denoise_with_patch_library(image, patch_size=8):
    padded, orig_h, orig_w = pad_to_multiple(image, patch_size)
    h, w, c = padded.shape
    output = np.zeros_like(padded)

    patches = split_into_patches(padded, patch_size, flatten=False)
    idx = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = patches[idx]
            idx += 1
            clean_patch = find_closest_patch(patch, noisy_library, clean_library)
            output[y:y+patch_size, x:x+patch_size, :] = clean_patch

    return output[:orig_h, :orig_w, :]

# ---------------------------
# Combined Pipeline Logic
# ---------------------------
def load_jump_fall_models(model_sets):
    models = {}
    for key, files in model_sets.items():
        with open(files[0], 'r') as f:
            model1_data = json.load(f)
        with open(files[1], 'r') as f:
            model2_data = json.load(f)

        models[key] = {
            'codebook': np.array(model1_data['codebook']),
            'patch_size': model1_data['patch_size'],
            'frame_size': tuple(model1_data['frame_size']),
            'num_textures': model1_data['num_textures'],
            'models': [[np.array(m['W1']), np.array(m['b1']),
                        np.array(m['W2']), np.array(m['b2'])]
                       for m in model1_data['models']],
            'weights2': [np.array(w) for w in model2_data['weights']],
            'biases2': [np.array(b) for b in model2_data['biases']],
        }
    return models

def run_pipeline(models, current_image, patch_size_denoiser=8, reshape_size=(28, 28)):
    codebook_centroids = models['codebook']
    patch_size = models['patch_size']
    frame_size = models['frame_size']
    num_textures = models['num_textures']
    dummy_codebook = DummyKMeans(codebook_centroids)

    grid_h = frame_size[0] // patch_size
    grid_w = frame_size[1] // patch_size
    num_cells = grid_h * grid_w

    # Step 1: Add small fixed Gaussian noise
    noise = np.random.normal(0, 0.01, current_image.shape)
    current_image_randomized = np.clip(current_image + noise, 0, 1)

    # Step 2: jump/fall.json
    current_ids = dummy_codebook.predict(split_into_patches(current_image_randomized, patch_size, flatten=True))
    next_ids = np.zeros(num_cells, dtype=np.int32)
    for idx in range(num_cells):
        neighbors = []
        y, x = divmod(idx, grid_w)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < grid_h and 0 <= nx < grid_w:
                neighbors.append(current_ids[ny*grid_w+nx])
            else:
                neighbors.append(current_ids[idx])
        counts = {j: 0 for j in range(num_textures)}
        current_id = current_ids[idx]
        counts[current_id] += 1
        for nid in neighbors:
            counts[nid] += 1
        max_count = max(counts.values())
        input_vec = np.zeros(num_textures*5)
        input_vec[0*num_textures + current_id] = counts[current_id] / max_count
        for j, nid in enumerate(neighbors):
            input_vec[(j+1)*num_textures + nid] = counts[nid] / max_count
        logits = forward_patch_model(input_vec, models['models'][idx])
        next_ids[idx] = np.argmax(logits)

    intermediate_frame_rgb = decode_frame(next_ids, patch_size, (grid_h, grid_w), codebook_centroids)

    # Step 3: jump2/fall2.json
    grayscale_image = get_brightness_values(intermediate_frame_rgb)
    pooled_image = max_pool(grayscale_image)
    final_features = forward_propagation(pooled_image, models['weights2'], models['biases2'])[-1]

    num_grid_y = reshape_size[0] // 2
    num_grid_x = reshape_size[1] // 2
    feature_size = 2*2
    final_features_reshaped = final_features.reshape(num_grid_y * num_grid_x, feature_size)
    jump2_output = reconstruct_image_from_features(final_features_reshaped, (2,2), reshape_size)

    # Step 4: Patch dataset denoising with padding
    jump2_rgb = np.stack([jump2_output]*3, axis=-1)
    cleaned_image = denoise_with_patch_library(jump2_rgb, patch_size=patch_size_denoiser)

    return jump2_output, cleaned_image

# ---------------------------
# Tkinter Interactive GUI
# ---------------------------
last_key = None

def interactive_pipeline(models, test_image_path, output_dir, patch_size_denoiser=8, reshape_size=(28,28)):
    os.makedirs(output_dir, exist_ok=True)

    current_image = load_and_preprocess_rgb(test_image_path, reshape_size)

    root = tk.Tk()
    root.title("Interactive Jump/Fall Pipeline")
    root.geometry("900x450")

    reconstructed_panel = tk.Label(root)
    reconstructed_panel.pack(side="left", padx=10, pady=10)

    counter = 1

    def update_images(command):
        nonlocal current_image, counter

        if command not in models:
            return

        jump2_output, cleaned = run_pipeline(models[command], current_image, patch_size_denoiser)

        cleaned_pil = Image.fromarray((cleaned*255).astype(np.uint8))

        # --- Draw jump/fall word inside the frame ---
        label = "jump" if command == 'w' else "fall"
        draw = ImageDraw.Draw(cleaned_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 8)
        except:
            font = ImageFont.load_default()
        draw.text((5, 5), label, fill=(255, 0, 0), font=font)

        cleaned_img_tk = ImageTk.PhotoImage(cleaned_pil.resize((200,200)))

        reconstructed_panel.configure(image=cleaned_img_tk)
        reconstructed_panel.image = cleaned_img_tk

        # Save with counter only, resized to 400x400
        cleaned_resized = cleaned_pil.resize((400, 400))
        cleaned_resized.save(os.path.join(output_dir, f"{counter}.png"))
        counter += 1

        # Feed only jump2 output forward
        current_image = np.stack([jump2_output]*3, axis=-1)

    def auto_press():
        global last_key
        last_w_press_time = time.time()
        while True:
            time.sleep(0.1)
            if last_key in ['w','s']:
                if time.time() - last_w_press_time >= 0.1:
                    update_images(last_key)
                    last_w_press_time = time.time()
                    last_key = None
            elif time.time() - last_w_press_time > 0.1:
                update_images('s')
                last_w_press_time = time.time()

    def on_key_press(event):
        global last_key
        key = event.char.lower()
        if key == 'q':
            root.destroy()
        elif key in models:
            last_key = key

    auto_press_thread = threading.Thread(target=auto_press, daemon=True)
    auto_press_thread.start()

    root.bind("<KeyPress>", on_key_press)
    root.mainloop()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    test_image_path = "flappytest.png"
    patch_size_denoiser = int(input("Enter patch size for denoiser (e.g. 8): "))
    model_sets = {
        'w': ["jump.json", "jump2.json"],
        's': ["fall.json", "fall2.json"]
    }
    models = load_jump_fall_models(model_sets)
    interactive_pipeline(models, test_image_path, "generated3", patch_size_denoiser)