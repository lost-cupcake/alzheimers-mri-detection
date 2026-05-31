import os
import sys
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from src.model.model_2d_cnn import Alzheimer2DCNN

MODEL_PATH = os.path.join(ROOT, "saved_models", "best_2d_cnn.pth")

# folder "0" = Alzheimer's, folder "1" = Normal  (index 0 = Alzheimer's)
CLASS_KEYS = ["alzheimers", "normal"]
LABELS = {"alzheimers": "Alzheimer's", "normal": "Normal"}

_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
_TF_DISPLAY = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

app = FastAPI(title="Alzheimer's MRI Classifier API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = Alzheimer2DCNN(num_classes=2)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        _model.eval()
    return _model


def _b64(arr_float):
    img = Image.fromarray((arr_float * 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    pil = Image.open(io.BytesIO(raw))

    model = get_model()
    activations, gradients = {}, {}
    
    target = model.features[12]
    h1 = target.register_forward_hook(lambda m, i, o: activations.update(value=o.clone()))
    h2 = target.register_full_backward_hook(lambda m, gi, go: gradients.update(value=go[0]))

    x = _TF(pil).unsqueeze(0)
    out = model(x)
    probs = F.softmax(out, dim=1).squeeze().detach().numpy()
    pred = int(out.argmax(1).item())

    model.zero_grad()
    out[0, pred].backward()
    grads = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((grads * activations["value"]).sum(1)).squeeze().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h1.remove(); h2.remove()

    base = _TF_DISPLAY(pil).permute(1, 2, 0).numpy()
    cam_img = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((128, 128))) / 255.0
    heat = cm.jet(cam_img)[..., :3]
    overlay = np.clip(0.55 * base + 0.45 * heat, 0, 1)

    return {
        "prediction": LABELS[CLASS_KEYS[pred]],
        "prediction_key": CLASS_KEYS[pred],
        "confidence": float(probs[pred]),
        "probabilities": [
            {"label": LABELS[CLASS_KEYS[i]], "key": CLASS_KEYS[i], "value": float(probs[i])}
            for i in range(len(CLASS_KEYS))
        ],
        "input_image": _b64(base),
        "gradcam_image": _b64(overlay),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)