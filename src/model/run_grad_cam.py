import os
import sys
import torch
import matplotlib.pyplot as plt

# Fix path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from src.model.model_2d_cnn import Alzheimer2DCNN
from src.model.grad_cam import GradCAM
from src.training.utils import get_slice_dataloaders

MODEL_PATH = os.path.join(ROOT, "saved_models", "best_2d_cnn.pth")
PROCESSED_SLICES_DIR = os.path.join(ROOT, "dataset", "OASIS1", "processed", "slices")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1️⃣ Load model
    model = Alzheimer2DCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(model)
    # 2️⃣ Load ONE validation MRI slice
    _, val_loader = get_slice_dataloaders(PROCESSED_SLICES_DIR, batch_size=1)
    imgs, labels = next(iter(val_loader))
    imgs = imgs.to(device)

    print("Input shape:", imgs.shape)  # should be (1, 1, H, W)

    # 3️⃣ Init Grad-CAM
    gradcam = GradCAM(
        model=model,
        target_layer = model.features[12]

  # 🔴 change if last conv name differs
    )

    # 4️⃣ Run Grad-CAM
    cam = gradcam(imgs)

    # 5️⃣ Plot
    mri_slice = imgs[0, 0].cpu()

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(mri_slice, cmap="gray")
    plt.title("MRI Slice")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mri_slice, cmap="gray")
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
