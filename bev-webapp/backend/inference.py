import torch
import numpy as np
from model import BEVModel


class BEVInference:
    def __init__(self, model_path, device):
        self.device = device
        self.model = BEVModel().to(device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.model.eval()

    def predict(self, input_tensor):
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            pred = self.model(input_tensor)

        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

        # binary BEV
        binary = (pred > 0.5).astype(np.uint8)

        return pred, binary