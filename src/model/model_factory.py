import torch
from src.model.model_2d_cnn import Alzheimer2DCNN
from src.model.resnet_model import (
    build_resnet18,
    build_efficientnet_b0,
    adapt_first_conv_to_grayscale
)

def get_model(model_name: str, num_classes: int = 2):
    model_name = model_name.lower()

    if model_name == "2d_cnn":
        return Alzheimer2DCNN(num_classes=num_classes)

    elif model_name == "resnet":
        model = build_resnet18(num_classes=num_classes, pretrained=True)
        model = adapt_first_conv_to_grayscale(model)
        return model

    elif model_name == "efficientnet":
        model = build_efficientnet_b0(num_classes=num_classes, pretrained=True)
        model = adapt_first_conv_to_grayscale(model)
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")
