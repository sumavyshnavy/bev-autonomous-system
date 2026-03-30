import cv2
import numpy as np
import torch


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception(f"Image not found: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = img / 255.0
    img = (img - 0.5) / 0.5

    return img


def get_edges(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges / 255.0


def preprocess(paths):
    images = []
    edges_all = []

    for p in paths:
        img = load_image(p)
        edge = get_edges(img * 0.5 + 0.5)

        images.append(img)
        edges_all.append(edge)

    # RGB stack → (H,W,9)
    images = np.stack(images)
    images = np.concatenate(images, axis=2)

    # edges → (H,W,3)
    edges_all = np.stack(edges_all)
    edges_all = np.transpose(edges_all, (1, 2, 0))

    h, w, _ = images.shape

    # positional encoding
    y_coords = np.linspace(0, 1, h).reshape(h, 1)
    y_coords = np.repeat(y_coords, w, axis=1)

    x_coords = np.linspace(0, 1, w).reshape(1, w)
    x_coords = np.repeat(x_coords, h, axis=0)

    pos = np.stack([x_coords, y_coords], axis=2)

    # final concat → (H,W,14)
    final = np.concatenate([images, edges_all, pos], axis=2)

    final = np.transpose(final, (2, 0, 1))  # (14,H,W)

    return torch.tensor(final, dtype=torch.float32).unsqueeze(0)