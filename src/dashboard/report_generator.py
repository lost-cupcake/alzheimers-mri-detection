
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_pdf_report(
    output_path: str,
    patient_id: str,
    prob_healthy: float,
    prob_ad: float,
    notes: str = ""
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 80, "Alzheimer MRI Detection Report")

    c.setFont("Helvetica", 12)
    y = height - 130
    c.drawString(50, y, f"Patient ID: {patient_id}")
    y -= 30
    c.drawString(50, y, f"Healthy probability: {prob_healthy:.3f}")
    y -= 20
    c.drawString(50, y, f"Alzheimer probability: {prob_ad:.3f}")
    y -= 40

    if notes:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Notes:")
        y -= 20
        c.setFont("Helvetica", 11)
        text_obj = c.beginText(50, y)
        for line in notes.split("\n"):
            text_obj.textLine(line)
        c.drawText(text_obj)

    c.showPage()
    c.save()
