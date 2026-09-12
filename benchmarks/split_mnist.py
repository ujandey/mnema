# Copyright (c) 2026 COGNX
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file at the repository root for the full terms.

import os
import urllib.request
import gzip
import numpy as np

def download_mnist(target_dir="data"):
    os.makedirs(target_dir, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lbl": "t10k-labels-idx1-ubyte.gz"
    }
    paths = {}
    for key, filename in files.items():
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        paths[key] = filepath
    return paths

def load_split_mnist(target_dir="data"):
    paths = download_mnist(target_dir)

    with gzip.open(paths["train_img"], "rb") as f:
        train_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784) / 255.0
    with gzip.open(paths["train_lbl"], "rb") as f:
        train_y = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(paths["test_img"], "rb") as f:
        test_x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784) / 255.0
    with gzip.open(paths["test_lbl"], "rb") as f:
        test_y = np.frombuffer(f.read(), np.uint8, offset=8)

    # Split into 5 sequential 2-class tasks
    tasks = []
    for task_id in range(5):
        c1, c2 = 2 * task_id, 2 * task_id + 1
        
        train_mask = (train_y == c1) | (train_y == c2)
        test_mask = (test_y == c1) | (test_y == c2)

        tasks.append({
            "task_id": task_id,
            "classes": (c1, c2),
            "train_x": train_x[train_mask],
            "train_y": train_y[train_mask],
            "test_x": test_x[test_mask],
            "test_y": test_y[test_mask]
        })
    return tasks