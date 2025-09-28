import os
import numpy as np
import json
from PIL import Image, ImageTk
import tkinter as tk
import threading
import time
from collections import deque # Use deque for an efficient fixed-size history buffer

# Track key state for continuous jump/fall
key_state = {'w': False}

# --- Utility Functions (unchanged, but included for completeness) ---

def max_pool(image, pool_size=(1, 2)):
    pooled_image = []
    height, width = image.shape
    pool_height, pool_width = pool_size

    for y in range(0, height, pool_height):
        for x in range(0, width, pool_width):
            # Ensure slicing stays within bounds (handles non-divisible dimensions gracefully)
            pool_region = image[y:min(y + pool_height, height), x:min(x + pool_width, width)]
            if pool_region.size > 0:
                 pooled_image.append(np.max(pool_region))
    return np.array(pooled_image).reshape(-1)

def forward_propagation(input_vector, weights, biases):
    # Ensure input_vector is 1D for dot product
    if input_vector.ndim > 1:
        input_vector = input_vector.flatten()
        
    activations = [input_vector]
    for i in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = np.maximum(0, z)
        activations.append(a)
    z = np.dot(activations[-1], weights[-1]) + biases[-1]
    activations.append(z)
    return activations

def reconstruct_image_from_features(features, grid_size, image_size):
    # This function needs the total number of features (grid cells)
    grid_height, grid_width = grid_size
    height, width = image_size
    
    # Calculate the size of the image that the features can fully reconstruct
    num_grid_y = height // grid_height
    num_grid_x = width // grid_width
    reconstructable_height = num_grid_y * grid_height
    reconstructable_width = num_grid_x * grid_width
    
    reconstructed_image = np.zeros((height, width))
    
    num_grids = num_grid_y * num_grid_x
    feature_size = grid_height * grid_width
    
    # Check if features is flat and slice/reshape it
    if features.size < num_grids * feature_size:
        print(f"Warning: Feature size {features.size} too small for grid {num_grids * feature_size}")
        return np.zeros((height, width))

    # Ensure features is correctly shaped for iteration (num_grids, feature_size)
    features_reshaped = features[:num_grids * feature_size].reshape((num_grids, feature_size))
    
    index = 0
    for y in range(num_grid_y):
        for x in range(num_grid_x):
            # grid is the feature vector for one grid cell
            grid = features_reshaped[index].reshape(grid_size)
            
            # Place the grid cell into the image
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
        
        # --- Crucial Addition: Load config to get num_input_frames ---
        config = model_data.get('config', {})
        num_input_frames = config.get('num_input_frames', 1) # Default to 1 if not found

        models[key] = {
            'weights': weights, 
            'biases': biases, 
            'num_input_frames': num_input_frames
        }
    return models

# --- Modified Interactive Function ---

def interactive_test_tk(models, test_image_path, grid_size, reshape_size):
    # Find the required sequence length from the models (assume they are consistent)
    if not models:
        print("Error: No models loaded.")
        return
        
    num_input_frames = next(iter(models.values()))['num_input_frames']
    
    # 1. Load and process the initial test image
    initial_image = Image.open(test_image_path).convert('L')
    initial_image = initial_image.resize(reshape_size)
    initial_image = np.array(initial_image, dtype=np.float32) / 255.0
    initial_pooled_feature = max_pool(initial_image)

    # 2. Initialize the History Buffer (deque)
    # The buffer stores the 'pooled' feature of the last N frames.
    # Start with N copies of the initial image's pooled feature.
    history_buffer = deque([initial_pooled_feature] * num_input_frames, maxlen=num_input_frames)
    
    # Set the current image (the last one in the buffer) for display/pooling in the next step
    current_image = initial_image

    # Tkinter setup
    root = tk.Tk()
    root.title(f"Interactive Model Test (Sequence N={num_input_frames})")
    root.geometry("500x600")

    # Display settings
    tk.Label(root, text=f"Sequence Length (N): {num_input_frames}").pack(pady=5)
    reconstructed_panel = tk.Label(root)
    reconstructed_panel.pack(pady=10)

    # Creativeness slider
    tk.Label(root, text="Creativeness (Noise)").pack()
    creativeness_scale = tk.Scale(root, from_=0.0, to=0.5, resolution=0.01, orient="horizontal")
    creativeness_scale.set(0.05) 
    creativeness_scale.pack(pady=5)
    
    tk.Label(root, text="Press 'W' to Jump, Release for Fall ('S')").pack(pady=5)


    def update_images(command):
        nonlocal current_image, history_buffer
        
        model_data = models.get(command)
        if not model_data:
            return
            
        weights = model_data['weights']
        biases = model_data['biases']

        # 1. Create the concatenated input vector from the history buffer
        input_vector = np.concatenate(history_buffer)

        # 2. Forward Propagation
        activations = forward_propagation(input_vector, weights, biases)
        output_features = activations[-1]

        # 3. Reconstruct and Add Noise
        num_grids = (reshape_size[0] // grid_size[0]) * (reshape_size[1] // grid_size[1])
        feature_size = grid_size[0] * grid_size[1]
        
        # Ensure output features match expected size before passing to reconstruct
        output_features = output_features[:num_grids * feature_size] 
        reconstructed_image = reconstruct_image_from_features(output_features, grid_size, reshape_size)

        noise_strength = creativeness_scale.get()
        noise = np.random.uniform(-noise_strength, noise_strength, reconstructed_image.shape).astype(np.float32)
        reconstructed_image = np.clip(reconstructed_image + noise, 0, 1)

        # 4. Display
        reconstructed_image_pil = Image.fromarray((reconstructed_image * 255).astype(np.uint8))
        reconstructed_img_tk = ImageTk.PhotoImage(reconstructed_image_pil.resize((450, 450), Image.Resampling.NEAREST)) # Use NEAREST for pixel art style

        reconstructed_panel.configure(image=reconstructed_img_tk)
        reconstructed_panel.image = reconstructed_img_tk

        # 5. Update for the Next Iteration
        # The reconstructed image is the 'next' frame.
        current_image = reconstructed_image 
        
        # Calculate the pooled feature of the *newly generated* frame
        new_pooled_feature = max_pool(current_image) 
        
        # Add the new pooled feature to the buffer (it automatically pushes out the oldest one)
        history_buffer.append(new_pooled_feature)


    def auto_press():
        while True:
            time.sleep(0.05) # Control the speed of the animation
            if key_state['w']:
                root.after(0, update_images, 'w')  # Schedule update on main thread
            else:
                root.after(0, update_images, 's')  # Schedule update on main thread

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

    # Start the simulation loop in a separate thread
    threading.Thread(target=auto_press, daemon=True).start()
    
    # Bind controls
    root.bind("<KeyPress>", on_key_press)
    root.bind("<KeyRelease>", on_key_release)
    
    # Start the Tkinter main loop
    root.mainloop()

# Script execution
if __name__ == "__main__":
    # Ensure you have flappytest.png and your trained model JSON files in the same directory
    test_image_path = "flappytest.png" 
    model_files = {
        'w': "jump.json", # Should contain config {'num_input_frames': N}
        's': "fall.json", # Should contain config {'num_input_frames': N}
    }
    grid_size = (2, 2)
    reshape_size = (28, 28)

    models = load_models(model_files)
    interactive_test_tk(models, test_image_path, grid_size, reshape_size)
