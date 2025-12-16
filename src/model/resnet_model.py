import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet18(num_classes=2, pretrained=True):
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes=2, pretrained=True):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model


def adapt_first_conv_to_grayscale(model):
    """
    Converts first Conv2d layer from 3-channel → 1-channel
    (required for MRI slices).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=(module.bias is not None),
            )

            with torch.no_grad():
                new_conv.weight[:] = module.weight.mean(dim=1, keepdim=True)
                if module.bias is not None:
                    new_conv.bias[:] = module.bias

            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            break

    return model
