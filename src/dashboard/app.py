# ==================== PATH FIX ====================
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

# ==================== STANDARD IMPORTS ====================
import io
import tempfile
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.cm as cm

# ==================== PROJECT IMPORTS ====================
from src.preprocessing.load_nifti import load_mri
from src.preprocessing.normalize import min_max_normalize
from src.model.model_2d_cnn import Alzheimer2DCNN
from src.model.grad_cam import GradCAM

# ==================== PDF ====================
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# ==================== CONFIG ====================
MODEL_PATH = os.path.join(ROOT, "saved_models", "best_2d_cnn.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Alzheimer MRI Detection", layout="wide")

# ==================== STYLE ====================
st.markdown("""
<style>
body { background:#0b1220; }
h1,h2,h3 { color:#f8fafc; font-weight:800; }
.block-container { padding-top:1.4rem; padding-bottom:2rem; }
.card {
  background:rgba(2,6,23,.85);
  border:1px solid rgba(148,163,184,.15);
  border-radius:16px;
  padding:16px;
}
.image-card {
  background:rgba(2,6,23,.85);
  border:1px solid rgba(148,163,184,.15);
  border-radius:16px;
  padding:12px;
}
.small { color:rgba(226,232,240,.75); font-size:0.92rem; }
.kpi { font-size:1.4rem; font-weight:900; margin:0; }
hr { border:none; height:1px; background:rgba(148,163,184,.15); margin:14px 0; }
</style>
""", unsafe_allow_html=True)

# ==================== HELPERS ====================
@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    model = Alzheimer2DCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def pil_from_slice(slice_2d, size=(128, 128)):
    slice_2d = np.clip(slice_2d, 0, 1)
    img = (slice_2d * 255).astype(np.uint8)
    return Image.fromarray(img, "L").resize(size)

def to_tensor(pil_img):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

def gradcam_overlay(base_pil, cam_2d, alpha=0.45):
    """
    base_pil : PIL Image (128x128)
    cam_2d   : numpy array (Hc x Wc), values [0,1]
    """
    # resize CAM to match base image size
    cam_img = Image.fromarray((cam_2d * 255).astype(np.uint8))
    cam_img = cam_img.resize(base_pil.size, resample=Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0

    # modern colormap API (avoids deprecation)
    cmap = matplotlib.colormaps.get_cmap("jet")
    heatmap = (cmap(cam_resized)[:, :, :3] * 255).astype(np.float32)

    base = np.array(base_pil.convert("RGB")).astype(np.float32)
    overlay = (1 - alpha) * base + alpha * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

def decision_from_probs(prob_normal, prob_ad, t_high=0.65):
    """
    Returns label + confidence style:
    - If one class prob >= t_high -> that label
    - Else -> Uncertain
    """
    if prob_ad >= t_high:
        return "Alzheimer", prob_ad
    if prob_normal >= t_high:
        return "Normal", prob_normal
    # uncertain: pick max but label as uncertain
    return "Uncertain", max(prob_normal, prob_ad)

def generate_pdf(
    filename,
    vol_shape,
    slice_idx,
    slice_probs,
    patient_probs,
    patient_label,
    t_high,
    mri_pil,
    overlay_pil
):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, h - 50, "Alzheimer MRI Detection Report")

    c.setFont("Helvetica", 10)
    c.drawString(40, h - 72, f"Generated: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}")
    c.drawString(40, h - 86, f"File: {filename}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, h - 120, "Patient-Level Summary")
    c.setFont("Helvetica", 11)
    c.drawString(40, h - 140, f"Assessment: {patient_label} (threshold={t_high:.2f})")
    c.drawString(40, h - 156, f"Patient Prob(Normal): {patient_probs[0]:.3f}")
    c.drawString(40, h - 172, f"Patient Prob(Alzheimer): {patient_probs[1]:.3f}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, h - 205, "Slice-Level (Selected Slice)")
    c.setFont("Helvetica", 11)
    c.drawString(40, h - 225, f"Slice Index: {slice_idx}")
    c.drawString(40, h - 241, f"Slice Prob(Normal): {slice_probs[0]:.3f}")
    c.drawString(40, h - 257, f"Slice Prob(Alzheimer): {slice_probs[1]:.3f}")
    c.drawString(40, h - 273, f"Volume Shape: {vol_shape}")

    # images
    m_buf, o_buf = io.BytesIO(), io.BytesIO()
    mri_pil.save(m_buf, "PNG"); overlay_pil.save(o_buf, "PNG")
    m_buf.seek(0); o_buf.seek(0)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, h - 310, "Imaging")
    c.setFont("Helvetica", 10)
    c.drawString(40, h - 326, "Left: MRI Slice | Right: Grad-CAM overlay (model attention)")

    img_w, img_h = 250, 250
    y = h - 600
    c.drawImage(ImageReader(m_buf), 40, y, width=img_w, height=img_h, mask='auto')
    c.drawImage(ImageReader(o_buf), 320, y, width=img_w, height=img_h, mask='auto')

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, 120, "Disclaimer")
    c.setFont("Helvetica", 10)
    disclaimer = (
        "This report is generated for educational/research purposes only. "
        "It is not a medical diagnosis and must not be used for clinical decision-making. "
        "Consult qualified medical professionals for interpretation."
    )
    text = c.beginText(40, 102)
    text.setLeading(14)
    words = disclaimer.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > 95:
            text.textLine(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        text.textLine(line)
    c.drawText(text)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# ==================== MAIN ====================
st.title("🧠 Alzheimer MRI Detection Dashboard")
st.caption("Slice viewer + Grad-CAM explainability + patient-level aggregation + PDF export")

model = load_model()
if model is None:
    st.error("Trained model not found. Please ensure saved_models/best_2d_cnn.pth exists.")
    st.stop()

# ---------- Controls ----------
st.sidebar.markdown("## Controls")
t_high = st.sidebar.slider("Decision threshold (Normal/Alzheimer)", 0.50, 0.95, 0.65, 0.01)
num_eval_slices = st.sidebar.slider("Patient-level slices to evaluate", 5, 60, 20, 1)
use_uniform_sampling = st.sidebar.checkbox("Uniform slice sampling (recommended)", value=True)
show_debug = st.sidebar.checkbox("Show slice-wise debug table", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("## Upload")
uploaded = st.file_uploader(
    "Upload MRI (.nii/.nii.gz OR Analyze .hdr + .img)",
    type=["nii", "nii.gz", "hdr", "img"],
    accept_multiple_files=True
)

if not uploaded:
    st.info("Upload MRI files to continue.")
    st.stop()

# ==================== LOAD VOLUME ====================
with tempfile.TemporaryDirectory() as tmp:
    paths = {}
    for f in uploaded:
        p = os.path.join(tmp, f.name)
        with open(p, "wb") as out:
            out.write(f.getbuffer())
        paths[f.name.lower()] = p

    nii = next((p for n, p in paths.items() if n.endswith(".nii") or n.endswith(".nii.gz")), None)

    if nii:
        volume = load_mri(nii)
        filename = os.path.basename(nii)
    else:
        hdr = next((p for n, p in paths.items() if n.endswith(".hdr")), None)
        img = next((p for n, p in paths.items() if n.endswith(".img")), None)
        if not (hdr and img):
            st.error("For Analyze format, upload BOTH .hdr and .img.")
            st.stop()
        volume = load_mri(hdr)
        filename = os.path.basename(hdr)

    volume = min_max_normalize(volume)
    H, W, Z = volume.shape

# ==================== SLICE VIEWER ====================
st.markdown("## Slice Viewer")
slice_idx = st.slider("Select MRI slice", 0, Z - 1, Z // 2, 1)

slice_2d = volume[:, :, slice_idx]
mri_pil = pil_from_slice(slice_2d, size=(128, 128))
img_tensor = to_tensor(mri_pil)

# ==================== SLICE-LEVEL INFERENCE ====================
with torch.no_grad():
    logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

prob_normal_slice = float(probs[0])
prob_ad_slice = float(probs[1])

slice_label, slice_conf = decision_from_probs(prob_normal_slice, prob_ad_slice, t_high=t_high)

# ==================== PATIENT-LEVEL AGGREGATION ====================
def sample_slice_indices(Z, k, uniform=True):
    if k >= Z:
        return list(range(Z))
    if uniform:
        # evenly spaced indices across volume
        return list(np.linspace(0, Z - 1, k).astype(int))
    # random sample but stable-ish
    rng = np.random.default_rng(42)
    return sorted(rng.choice(np.arange(Z), size=k, replace=False).tolist())

eval_indices = sample_slice_indices(Z, num_eval_slices, uniform=use_uniform_sampling)

slice_preds = []
slice_probs_list = []

with torch.no_grad():
    for idx in eval_indices:
        pil = pil_from_slice(volume[:, :, idx], size=(128, 128))
        t = to_tensor(pil)
        p = F.softmax(model(t), dim=1)[0].cpu().numpy()
        pn, pa = float(p[0]), float(p[1])

        # store
        slice_probs_list.append((idx, pn, pa))
        # hard prediction by threshold logic (normal/ad/uncertain)
        lbl, conf = decision_from_probs(pn, pa, t_high=t_high)
        slice_preds.append(lbl)

# patient-level probability (mean across evaluated slices)
pn_mean = float(np.mean([x[1] for x in slice_probs_list]))
pa_mean = float(np.mean([x[2] for x in slice_probs_list]))

patient_label, patient_conf = decision_from_probs(pn_mean, pa_mean, t_high=t_high)

# majority vote (ignoring "Uncertain" if possible)
def majority_vote(labels):
    # prefer decisive labels
    decisive = [l for l in labels if l in ("Normal", "Alzheimer")]
    if len(decisive) == 0:
        return "Uncertain"
    return max(set(decisive), key=decisive.count)

patient_vote = majority_vote(slice_preds)

# ==================== GRAD-CAM (on selected slice) ====================
# Use last Conv2d in your features: index 12 in your model printout (Conv2d 128->256)
target_layer = model.features[12]
cam = GradCAM(model, target_layer)(img_tensor)
overlay = gradcam_overlay(mri_pil, cam, alpha=0.45)

# ==================== UI LAYOUT ====================
c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("<div class='image-card'>", unsafe_allow_html=True)
    st.subheader("MRI Slice")
    st.image(mri_pil, width=360)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='image-card'>", unsafe_allow_html=True)
    st.subheader("Grad-CAM")
    st.image(overlay, width=360)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== SUMMARY (balanced confidence UI) ====================
st.markdown("## Patient Summary")

s1, s2, s3 = st.columns([1.2, 1, 1])

with s1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"**File:** {filename}")
    st.markdown(f"**Volume Shape:** {(H, W, Z)}")
    st.markdown(f"**Selected Slice:** {slice_idx}")
    st.markdown(f"**Evaluated Slices for Patient-Level:** {len(eval_indices)}")
    st.markdown("</div>", unsafe_allow_html=True)

with s2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Slice-level prediction**")
    st.markdown(f"<p class='kpi'>{slice_label}</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Prob(Normal): {prob_normal_slice:.3f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Prob(Alzheimer): {prob_ad_slice:.3f}</div>", unsafe_allow_html=True)
    st.progress(min(prob_ad_slice, 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

with s3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Patient-level prediction** (mean prob + threshold)")
    st.markdown(f"<p class='kpi'>{patient_label}</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Mean Prob(Normal): {pn_mean:.3f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Mean Prob(Alzheimer): {pa_mean:.3f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Majority vote (decisive slices): <b>{patient_vote}</b></div>", unsafe_allow_html=True)
    st.progress(min(pa_mean, 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

# Guidance message
if patient_label == "Uncertain":
    st.warning(
        f"Patient-level result is **Uncertain** at threshold={t_high:.2f}. "
        f"Try lowering threshold slightly (e.g., 0.60) or increase evaluated slices."
    )

# ==================== DEBUG TABLE ====================
if show_debug:
    st.markdown("## Slice-wise Debug")
    st.caption("This helps you verify that the model can output Normal/Alzheimer across slices and subjects.")

    # sort by Alzheimer prob desc
    sorted_rows = sorted(slice_probs_list, key=lambda x: x[2], reverse=True)

    # build a small table
    debug = []
    for idx, pn, pa in sorted_rows:
        lbl, conf = decision_from_probs(pn, pa, t_high=t_high)
        debug.append({
            "slice": idx,
            "prob_normal": round(pn, 4),
            "prob_alzheimer": round(pa, 4),
            "label": lbl
        })

    st.dataframe(debug, use_container_width=True)

# ==================== PDF EXPORT ====================
st.markdown("## Export")

pdf_bytes = generate_pdf(
    filename=filename,
    vol_shape=(H, W, Z),
    slice_idx=slice_idx,
    slice_probs=(prob_normal_slice, prob_ad_slice),
    patient_probs=(pn_mean, pa_mean),
    patient_label=patient_label,
    t_high=t_high,
    mri_pil=mri_pil,
    overlay_pil=overlay
)

st.download_button(
    "📄 Download PDF Report",
    data=pdf_bytes,
    file_name=f"alz_report_{os.path.splitext(filename)[0]}_slice{slice_idx}.pdf",
    mime="application/pdf"
)

st.caption("Note: This tool is for educational/research use only and is not a clinical diagnosis system." \
"\nMade By Rohit kapoor")
