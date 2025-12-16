🧠 Alzheimer’s MRI Detection using CNN & Grad-CAM

An end-to-end deep learning system for Alzheimer’s disease detection from brain MRI scans, built using PyTorch and Streamlit, with a strong focus on explainable AI through Grad-CAM visualizations.

🚀 Project Overview

Magnetic Resonance Imaging (MRI) scans are inherently 3D volumetric data, making direct modeling computationally expensive and less interpretable.
This project adopts a slice-based learning approach, converting 3D MRI volumes into 2D slices and applying a 2D Convolutional Neural Network (CNN) for classification.

To improve transparency and trust, Grad-CAM is used to highlight brain regions that most influence the model’s predictions.

An interactive Streamlit dashboard allows users to upload MRI scans, visualize predictions, and inspect Grad-CAM heatmaps.

🧩 Key Features

📊 Slice-based learning from 3D MRI volumes

🧠 2D CNN for Alzheimer vs Normal classification

🧮 Patient-level prediction via slice aggregation

🔍 Grad-CAM based explainability

🖥️ Interactive Streamlit dashboard

📄 Automated prediction report generation

⚖️ Ethical handling of medical data (no data pushed to GitHub)

🗂 Dataset

Dataset Used: OASIS (Open Access Series of Imaging Studies)

Data Type: 3D MRI brain scans (.nii, .hdr, .img)

Labels: Normal / Alzheimer’s Disease

⚠️ Note:
Medical data is not included in this repository for ethical and legal reasons.

🔬 Methodology
1️⃣ MRI Preprocessing & Slicing

Each 3D MRI volume is sliced along the axial plane

Central and informative slices are selected

Intensity normalization is applied

2️⃣ Slice-Level Prediction

Each slice is independently classified using a 2D CNN

Model outputs probabilities for:

Normal

Alzheimer’s Disease

3️⃣ Patient-Level Aggregation

Slice-level predictions are aggregated using:

Mean probability across slices

Majority voting

Confidence thresholds

This yields a patient-level prediction rather than relying on a single slice.

4️⃣ Explainability with Grad-CAM

Grad-CAM generates heatmaps from the final convolutional layers

Highlights brain regions influencing predictions

Improves interpretability and transparency

🧠 Model Architecture

Input: 128 × 128 grayscale MRI slice

Convolutional blocks with BatchNorm & ReLU

Adaptive pooling for spatial consistency

Fully connected classifier

Output: Probability scores for each class

🖥️ Dashboard Preview
Dashboard Interface

Interactive MRI upload, slice navigation, and prediction display.

Grad-CAM Visualization

Heatmap showing regions contributing to Alzheimer’s prediction.

🛠 Tech Stack

Deep Learning: PyTorch

Explainability: Grad-CAM

Data Processing: NumPy, NiBabel

Dashboard: Streamlit

Visualization: Matplotlib, PIL

▶️ How to Run Locally
# Clone repository
git clone https://github.com/lost-cupcake/alzheimers-mri-detection.git
cd alzheimers-mri-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run src/dashboard/app.py

⚠️ Disclaimer

This project is intended strictly for educational and research purposes.
It is NOT a medical device and must NOT be used for clinical diagnosis or treatment.

📌 Future Improvements

3D CNN or hybrid 2D–3D modeling

Multi-class Alzheimer staging

Radiologist-in-the-loop validation

Model calibration & uncertainty estimation

Regulatory-grade evaluation pipelines

👨‍💻 Author

GitHub: lost-cupcake
