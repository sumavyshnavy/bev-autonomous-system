import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from model import BEVModel
from risk import generate_risk
from planner import plan_path

# ------------------ SETUP ------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "models/best_bev_model.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ LOAD MODEL ------------------
try:
    print("🔄 Loading model...")
    model = BEVModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ MODEL LOAD ERROR:", e)


# ------------------ PREPROCESS ------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found: {path}")
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

    images = np.stack(images)
    images = np.concatenate(images, axis=2)

    edges_all = np.stack(edges_all)
    edges_all = np.transpose(edges_all, (1, 2, 0))

    h, w, _ = images.shape

    y_coords = np.linspace(0, 1, h).reshape(h, 1)
    y_coords = np.repeat(y_coords, w, axis=1)

    x_coords = np.linspace(0, 1, w).reshape(1, w)
    x_coords = np.repeat(x_coords, h, axis=0)

    pos = np.stack([x_coords, y_coords], axis=2)

    final = np.concatenate([images, edges_all, pos], axis=2)
    final = np.transpose(final, (2, 0, 1))

    return torch.tensor(final, dtype=torch.float32).unsqueeze(0)


# ------------------ CORS ------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


# ------------------ API ------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🔥 Request received")

        files = request.files
        destination = request.form.get("destination", "forward")

        paths = []
        for key in ["img1", "img2", "img3"]:
            if key not in files:
                raise ValueError(f"{key} not uploaded")

            file = files[key]
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            paths.append(path)

        print("✅ Images saved:", paths)

        # ---------- MODEL ----------
        input_tensor = preprocess(paths).to(device)

        with torch.no_grad():
            pred = model(input_tensor)

        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

        # -------- TEMPORAL SMOOTHING --------
        pred = cv2.GaussianBlur(pred, (9, 9), 0)

        # -------- CONFIDENCE MAP --------
        confidence = np.power(pred, 0.5)
        confidence = cv2.normalize(confidence, None, 0, 255, cv2.NORM_MINMAX)
        confidence = confidence.astype(np.uint8)
        confidence_color = cv2.applyColorMap(confidence, cv2.COLORMAP_VIRIDIS)

        confidence_path = os.path.join(OUTPUT_FOLDER, "confidence.png")
        cv2.imwrite(confidence_path, confidence_color)

        # ---------- BEV ----------
        h, w = pred.shape

        distance_weight = np.linspace(1.5, 0.5, h).reshape(h, 1)
        weighted_pred = pred * distance_weight

        bev = (weighted_pred > 0.3).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        bev = cv2.dilate(bev, kernel, iterations=2)

        bev_path = os.path.join(OUTPUT_FOLDER, "bev.png")
        cv2.imwrite(bev_path, bev * 255)

        # ---------- RISK ----------
        risk = generate_risk(bev)
        risk = cv2.GaussianBlur(risk, (11, 11), 0)

        risk_uint8 = (risk * 255).astype(np.uint8)
        risk_color = cv2.applyColorMap(risk_uint8, cv2.COLORMAP_JET)
        risk_path = os.path.join(OUTPUT_FOLDER, "risk.png")
        cv2.imwrite(risk_path, risk_color)

        # ---------- TRAJECTORY ----------
        inflated_bev = cv2.dilate(bev.astype(np.uint8), np.ones((15, 15), np.uint8))
        path_pts = plan_path(inflated_bev, risk, destination)

        if not path_pts:
            center_x, center_y = h // 2, w // 2
            path_pts = [(center_x - i, center_y) for i in range(50) if center_x - i >= 0]

        vis = np.stack([bev * 255]*3, axis=2)

        # GRID
        grid_size = 20
        for i in range(0, h, grid_size):
            cv2.line(vis, (0, i), (w, i), (50, 50, 50), 1)
        for j in range(0, w, grid_size):
            cv2.line(vis, (j, 0), (j, h), (50, 50, 50), 1)

        # EGO
        ego_x, ego_y = h // 2, w // 2
        cv2.circle(vis, (ego_y, ego_x), 5, (0, 255, 0), -1)

        # TRAJECTORY
        for (i, j) in path_pts:
            vis[i, j] = [0, 0, 255]

        if path_pts:
            end = path_pts[-1]
            vis[end[0], end[1]] = [255, 0, 0]

        traj_path = os.path.join(OUTPUT_FOLDER, "trajectory.png")
        cv2.imwrite(traj_path, vis)

        base_url = request.host_url.rstrip("/")

        print("✅ Outputs generated successfully")

        return jsonify({
            "bev": f"{base_url}/outputs/bev.png",
            "risk": f"{base_url}/outputs/risk.png",
            "trajectory": f"{base_url}/outputs/trajectory.png",
            "confidence": f"{base_url}/outputs/confidence.png"
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)})


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)