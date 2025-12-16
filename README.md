🧠 Alzheimer’s Detection from MRI using 2D CNN with Explainable AI (Grad-CAM)
Overview

This project implements an end-to-end deep learning system for detecting Alzheimer’s disease patterns from brain MRI scans using a 2D Convolutional Neural Network (CNN).
To improve trust and interpretability, the system integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which regions of the MRI influenced the model’s predictions.



&nbsp;🖥️ Dashboard Overview

Interactive Streamlit dashboard for MRI upload, slice navigation, and patient-level prediction.



!\[Dashboard UI](assets/dashboard\_ui.png)



&nbsp;🔍 Model Output \& Explainability

Grad-CAM visualization highlighting brain regions influencing Alzheimer’s prediction.



!\[Grad-CAM Output](assets/model\_output\_gradcam.png)



⚠️ Disclaimer:
This project is intended for educational and research purposes only. It is not a clinically certified medical system and must not be used for real-world medical diagnosis.

Motivation

Alzheimer’s disease is a progressive neurodegenerative disorder where early detection can significantly impact patient care.
While deep learning models can achieve high accuracy, they often act as black boxes. In medical imaging, explainability is critical.

This project focuses on:

Building a technically correct ML pipeline

Providing model explainability

Designing a realistic, hospital-style decision logic (confidence thresholds, uncertainty handling)

Dataset

OASIS (Open Access Series of Imaging Studies)

MRI volumes with clinical labels derived from CDR (Clinical Dementia Rating)

CDR = 0 → Normal

CDR > 0 → Alzheimer / Dementia

Each MRI scan is a 3D volume (e.g., 256 × 256 × 160) consisting of multiple 2D brain slices.

Why a 2D CNN?

Instead of using a computationally expensive 3D CNN, this project uses a 2D CNN trained on individual MRI slices.

Key reasons:

Lower computational cost

Faster experimentation

Common approach in medical imaging when slice-level annotations are unavailable

Training strategy (weak supervision):

Each slice inherits the patient-level diagnosis

The model learns statistical structural patterns across many slices and patients

Model Architecture

Input: 128 × 128 grayscale MRI slice

Convolutional blocks with BatchNorm and ReLU

Progressive spatial downsampling

Fully connected classifier with dropout

Output:

Probability of Normal

Probability of Alzheimer

Loss function: Cross-Entropy Loss
Optimizer: Adam

Inference Pipeline

Load MRI volume (.nii, .nii.gz, or .hdr + .img)

Normalize voxel intensities

Extract multiple 2D slices from the volume

Run each slice through the trained 2D CNN

Obtain slice-level probabilities:

Prob(Normal), Prob(Alzheimer)

Patient-Level Aggregation (Important)

A single MRI slice is not sufficient for diagnosis.

To address this:

The system evaluates N slices (default = 20) uniformly sampled across the brain

Aggregates predictions using:

Mean probability across slices

Majority voting across confident slices

This mimics how radiologists scroll through MRI volumes.

Confidence Threshold \& Uncertainty Handling

A confidence threshold (default = 0.65) is applied symmetrically to both classes.

Decision logic:
If Prob(Alzheimer) ≥ threshold → Alzheimer
Else if Prob(Normal) ≥ threshold → Normal
Else → Uncertain

Why this matters:

Prevents forced predictions

Reduces false positives

Reflects real-world medical AI safety practices

Explainability with Grad-CAM
What Grad-CAM does:

Grad-CAM visualizes which regions of the MRI slice most influenced the model’s prediction.

How it works (high-level):

Computes gradients of the target class score (e.g., Alzheimer)

Weights feature maps from the final convolution layer

Produces a coarse heatmap highlighting influential regions

Important notes:

Grad-CAM does not identify medical biomarkers

It shows model attention, not clinical localization

Blurry heatmaps are expected and correct (deep-layer semantics)

Dashboard Features

Built using Streamlit, the dashboard provides:

MRI slice viewer with slider

Grad-CAM overlay visualization

Slice-level prediction with probabilities

Patient-level prediction (mean + majority vote)

Adjustable confidence threshold

Uncertainty-aware outputs (Normal / Alzheimer / Uncertain)

PDF report export for documentation

Evaluation

High validation performance on OASIS dataset

ROC-AUC ≈ 1.0 on internal validation

⚠️ Note:
High ROC-AUC indicates strong dataset-level separability, not guaranteed real-world performance.

Limitations

Slice-level training with patient-level labels (weak supervision)

No longitudinal or multi-modal clinical data

Not clinically validated

No regulatory approval (FDA / CDSCO / CE)

Future Work

Patient-level 3D CNN or 2.5D multi-slice models

Multi-class staging (Normal / MCI / Alzheimer)

External validation on ADNI dataset

Clinical collaboration with neurologists/radiologists

Regulatory-compliant evaluation pipeline

Tech Stack

Python

PyTorch

NumPy

Nibabel

Matplotlib

Streamlit

ReportLab (PDF export)

Final Note

This project demonstrates:

Strong ML system design

Realistic medical AI decision logic

Explainable AI integration

Production-style debugging and UI thinking

It is company-level and resume-worthy, while being responsibly framed as a research and educational system.

