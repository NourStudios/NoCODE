import os
import numpy as np
import json
from PIL import Image
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_brightness_values(image_path, reshape_size):
    image = Image.open(image_path).convert('L').resize(reshape_size)
    arr = np.array(image, dtype=np.float32) / 255.0
    return arr

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

class AdamOpt:
    def __init__(self, weights, biases, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, clip_value=1.0, decay=0.9995):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.decay = decay
        self.t = 0
        self.clip_value = clip_value
        self.m_w = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.v_w = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.m_b = [np.zeros_like(b, dtype=np.float32) for b in biases]
        self.v_b = [np.zeros_like(b, dtype=np.float32) for b in biases]

    def update(self, weights, biases, grads_w, grads_b):
        self.t += 1
        self.lr *= self.decay  # decay LR slowly
        for i in range(len(weights)):
            gw = np.array(grads_w[i], dtype=np.float32)
            gb = np.array(grads_b[i], dtype=np.float32)
            gw = np.clip(gw, -self.clip_value, self.clip_value)
            gb = np.clip(gb, -self.clip_value, self.clip_value)

            # weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gw
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gw ** 2)
            m_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            weights[i] -= self.lr * m_hat / (np.sqrt(v_hat + self.eps))

            # biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gb
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gb ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b + self.eps))

def initialize_weights_and_biases(input_size, hidden_sizes, output_size):
    weights, biases = [], []
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
        b = np.zeros(layer_sizes[i+1], dtype=np.float32)
        weights.append(w.astype(np.float32))
        biases.append(b)
    return weights, biases

def relu(x): return np.maximum(0, x)

def forward(inputs, weights, biases):
    acts = [inputs]
    for i in range(len(weights)-1):
        z = acts[-1] @ weights[i] + biases[i]
        a = relu(z)
        acts.append(a)
    z = acts[-1] @ weights[-1] + biases[-1]
    acts.append(z)
    return acts

def backward(acts, weights, target):
    batch_size = target.shape[0]
    deltas = [acts[-1] - target]
    for i in range(len(weights)-2, -1, -1):
        delta = deltas[0] @ weights[i+1].T
        delta *= (acts[i+1] > 0)
        deltas.insert(0, delta)
    grads_w, grads_b = [], []
    for i in range(len(weights)):
        grads_w.append(acts[i].T @ deltas[i] / batch_size)
        grads_b.append(np.mean(deltas[i], axis=0))
    return grads_w, grads_b

def patch_consistency(inputs, targets):
    """
    Measures how consistent patches are.
    High variance across same-neighbor inputs = global dependent.
    Low variance = neighbor dependent.
    """
    mean_t = np.mean(targets, axis=0)
    var_t = np.mean((targets - mean_t) ** 2, axis=0)
    return var_t / (np.max(var_t) + 1e-8)  # normalize [0..1]

# Training
def train_model(data_folder, model_file, reshape_size, grid_sizes, hidden_sizes, lr, iterations, num_input_frames, batch_size=32):
    file_list = sorted(os.listdir(data_folder), key=natural_sort_key)
    processed = []

    # Preprocess frames
    for f in file_list:
        if f.lower().endswith(('.png','.jpg','.jpeg')):
            arr = get_brightness_values(os.path.join(data_folder, f), reshape_size)
            pooled = max_pool(arr)
            grid = extract_grid_features(arr, grid_sizes)
            processed.append({'pooled': pooled, 'grid': grid})

    # Build input/target sequences
    inputs, targets = [], []
    for i in range(1, len(processed), 2):
        if i - num_input_frames < 0:
            continue
        seq = [processed[j]['pooled'] for j in range(i - num_input_frames, i)]
        inputs.append(np.concatenate(seq))
        targets.append(processed[i]['grid'])

    inputs, targets = np.array(inputs, np.float32), np.array(targets, np.float32)
    input_size, output_size = inputs.shape[1], targets.shape[1]
    weights, biases = initialize_weights_and_biases(input_size, hidden_sizes, output_size)
    opt = AdamOpt(weights, biases, lr)

    def compute_patch_dependency(inputs, targets):
        """
        Returns weights per output patch (0=global, 1=local)
        """
        from collections import defaultdict

        patch_dict = defaultdict(list)
        for i in range(len(inputs)):
            key = tuple(inputs[i])  # convert input to hashable
            patch_dict[key].append(targets[i])

        dependency_weights = np.zeros(targets.shape[1], dtype=np.float32)
        for patch_input, t_list in patch_dict.items():
            t_arr = np.array(t_list)
            var = np.mean((t_arr - np.mean(t_arr, axis=0))**2, axis=0)
            dependency_weights += (1 - var / (np.max(var) + 1e-8))  # high weight = local

        dependency_weights /= np.max(dependency_weights)  # normalize to [0,1]
        return dependency_weights

    dependency_weights = compute_patch_dependency(inputs, targets)

    for it in range(iterations):
        idx = np.random.choice(len(inputs), batch_size, replace=False)
        x, y = inputs[idx], targets[idx]

        acts = forward(x, weights, biases)
        pred = acts[-1]

        # weighted loss: local/global patch weighting
        w_loss = dependency_weights * ((pred - y) ** 2)
        loss = np.mean(w_loss)

        grads_w, grads_b = backward(acts, weights, y)
        opt.update(weights, biases, grads_w, grads_b)

        if it % 1 == 0 or it == iterations-1:
            print(f"Iter {it}, Loss {loss:.6f}, LR {opt.lr:.6e}")

    model = {
        'weights': [w.tolist() for w in weights],
        'biases': [b.tolist() for b in biases],
        'config': {
            'reshape_size': reshape_size,
            'grid_sizes': grid_sizes,
            'num_input_frames': num_input_frames
        }
    }
    with open(model_file, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Model saved to {model_file}")


if __name__ == "__main__":
    data_folder = "jump"                     # folder with your frames
    model_file = "jump_sequence.json"        # where to save the trained model
    reshape_size = (40, 40)                  # size each frame is resized to
    grid_sizes = [2, 2]                      # patch size for grid features
    hidden_sizes = [1000, 1000]              # hidden layer sizes
    learning_rate = 0.001                     # initial learning rate
    iterations = 2000                        # training iterations
    num_input_frames = 1                # number of preceding frames used as input
    batch_size = 32                           # mini-batch size for training

    train_model(
        data_folder=data_folder,
        model_file=model_file,
        reshape_size=reshape_size,
        grid_sizes=grid_sizes,
        hidden_sizes=hidden_sizes,
        lr=learning_rate,
        iterations=iterations,
        num_input_frames=num_input_frames,
        batch_size=batch_size
    )

