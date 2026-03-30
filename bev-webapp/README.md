# BEV + Risk + Trajectory Planning System

This project implements a simplified perception-to-planning pipeline for autonomous driving. Given multiple camera inputs, the system generates a Bird’s Eye View (BEV) representation, estimates risk across the scene, and computes a safe trajectory based on the environment.

---

## Overview

The goal of this project is to simulate how an autonomous vehicle understands its surroundings and makes navigation decisions.

The pipeline consists of:

* Processing multiple camera views
* Converting them into a unified top-down (BEV) representation
* Identifying potentially risky regions
* Planning a path that avoids obstacles and high-risk zones

---

## Features

### BEV Occupancy Mapping

* Combines three input images into a spatial top-down view
* Highlights obstacles while reducing noise
* Serves as the base representation for further analysis

### Risk Map Generation

* Produces a heatmap indicating unsafe regions
* Higher intensity corresponds to higher risk
* Helps guide safe navigation decisions

### Trajectory Planning

* Computes a path from the vehicle’s position to a target direction
* Avoids obstacles and high-risk regions
* Supports directional goals: forward, left, and right

### Additional Enhancements

* Web-based interface for easy interaction
* Confidence map to visualize model certainty
* Grid-based visualization for better interpretability
* End-to-end pipeline from input images to planning output

---

## Demo

The system allows users to:

1. Upload three driving scene images
2. Select a goal direction
3. Run the model

The outputs include:

* BEV Occupancy Map
* Risk Map
* Planned Trajectory
* Confidence Map

---

## Model Weights

The trained model is not included in the repository due to size constraints.

Download it here:
https://drive.google.com/file/d/1pktY-X2p0nOwkDqlQ2f-A-u8Bl6njBby/view?usp=sharing

Place the file inside:

backend/models/

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/bev-autonomous-system.git
cd bev-autonomous-system
pip install -r requirements.txt
```

---

## Running the Project

### Start the backend

```bash
cd backend
python app.py
```

### Start the frontend

```bash
cd frontend
python -m http.server 8000
```

Then open:

http://127.0.0.1:8000

---

## Input Requirements

Provide three images representing different views of a driving scene. These can be approximate front/side views; the system is designed to process them jointly.

---

## Outputs

* BEV occupancy map
* Risk heatmap
* Planned trajectory
* Confidence visualization

---

## Tech Stack

* Python
* PyTorch
* OpenCV
* Flask
* HTML, CSS, JavaScript

---

## Future Work

* Extending to real-time video input
* Improving path planning with advanced algorithms (e.g., A*, RL-based methods)
* Incorporating multi-object tracking
* Integrating additional sensors such as LiDAR

---

## Authors

Team Loopers:
Arunabho Das
Sreeparvathy
Suma Vyshnavy
Neha Salapu

---

## Notes

This project focuses on demonstrating the integration of perception and planning in a simplified autonomous driving setup, rather than achieving production-level accuracy.
