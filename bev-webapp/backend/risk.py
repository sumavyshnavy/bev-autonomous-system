import numpy as np
import cv2


def generate_risk(binary):
    obstacle = (binary * 255).astype(np.uint8)

    # vehicle size expansion
    kernel = np.ones((15, 15), np.uint8)
    expanded = cv2.dilate(obstacle, kernel, iterations=2)
    safe_obstacle = expanded / 255.0
    free_space = 1 - safe_obstacle

    h, w = binary.shape
    cy, cx = h // 2, w // 2

    risk = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)

            # smoother decay
            dist_weight = np.exp(-dist / 50.0)

            risk[i, j] = (
                0.6 * safe_obstacle[i, j] +
                0.4 * (1 - free_space[i, j])
            ) * dist_weight

    risk = cv2.GaussianBlur(risk, (11, 11), 0)

    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-6)

    return risk