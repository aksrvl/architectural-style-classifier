import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
NUM_CLASSES   = 24

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

device = torch.device("cpu")

def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    return model.to(device)


def load_model(model_name: str, weights_path: str) -> nn.Module:
    model= build_model(model_name, NUM_CLASSES)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(image_bytes: bytes, model: nn.Module, class_names: list, top_k: int = 3) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    return {
        "top_predictions": [
            {"style": class_names[idx], "confidence": round(float(probabilities[idx]) * 100, 2)}
            for idx in top_indices
        ]
    }