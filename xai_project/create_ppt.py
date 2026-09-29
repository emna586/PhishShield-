from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_presentation():
    prs = Presentation()
    
    # Theme settings
    title_color = RGBColor(0, 51, 102)

    # Slide 1: Title
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Phishing Detection Using Artificial Intelligence"
    subtitle.text = "Integrating Explainable AI and Adversarial Robustness\n\nFinal Defense Presentation"

    # Slide 2: Context & Problem
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Context & Problem Statement"
    tf = slide.placeholders[1].text_frame
    tf.text = "Cyber threats are evolving rapidly, with phishing being a primary attack vector."
    p = tf.add_paragraph()
    p.text = "Machine Learning offers great accuracy, but poses two major challenges in cybersecurity:"
    p.level = 1
    p = tf.add_paragraph()
    p.text = "The Black Box Problem: Lack of transparency in AI decisions prevents trust from SOC analysts."
    p.level = 2
    p = tf.add_paragraph()
    p.text = "Adversarial Vulnerability: Hackers can trick models with minor, mathematically crafted modifications."
    p.level = 2

    # Slide 3: Methodology
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Methodology & Data Pipeline"
    tf = slide.placeholders[1].text_frame
    tf.text = "Dataset: PhishFOE (101,063 samples, 36 extracted features)"
    p = tf.add_paragraph()
    p.text = "Preprocessing Pipeline:"
    p.level = 0
    p = tf.add_paragraph()
    p.text = "Data Cleaning & Normalization (StandardScaler)"
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Class Balancing: SMOTE to prevent bias towards legitimate URLs"
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Feature Selection: ANOVA and Recursive Feature Elimination (RFE)"
    p.level = 1

    # Slide 4: ML Results
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Machine Learning Performance (Bootstrap CI)"
    
    # Remove text placeholder to put table
    sp = slide.shapes[1]
    sp.element.getparent().remove(sp.element)
    
    rows = 4
    cols = 6
    left = Inches(0.5)
    top = Inches(2.0)
    width = Inches(9.0)
    height = Inches(1.5)
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    
    # Adjust column widths
    table.columns[0].width = Inches(1.8)
    table.columns[1].width = Inches(1.4)
    table.columns[2].width = Inches(1.4)
    table.columns[3].width = Inches(1.4)
    table.columns[4].width = Inches(1.4)
    table.columns[5].width = Inches(1.6)

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Inference"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    data = [
        ["Random Forest", "98.78%", "98.70%", "98.81%", "98.75%", "12.4 ms"],
        ["XGBoost", "98.80%", "98.75%", "98.85%", "98.80%", "3.8 ms"],
        ["LightGBM", "98.86%", "98.82%", "98.90%", "98.86%", "2.1 ms"]
    ]
    for row_idx, row_data in enumerate(data):
        for col_idx, cell_data in enumerate(row_data):
            table.cell(row_idx + 1, col_idx).text = cell_data

    # Slide 5: XAI
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Explainable AI (XAI) - Triangular Validation"
    
    sp = slide.shapes[1]
    sp.element.getparent().remove(sp.element)
    
    rows = 6
    cols = 4
    table = slide.shapes.add_table(rows, cols, left, Inches(1.8), width, Inches(2.5)).table
    
    headers = ["Rank", "SHAP (Global)", "PFI (Permutation)", "LIME (Local Avg)"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    data = [
        ["1", "URL_Length", "URL_Length", "URL_Length"],
        ["2", "Nb_Special_Chars", "Nb_Special_Chars", "Nb_Special_Chars"],
        ["3", "HTTPS_Token", "HTTPS_Token", "Domain_Age"],
        ["4", "Domain_Age", "Subdomain_Count", "HTTPS_Token"],
        ["5", "Subdomain_Count", "Domain_Age", "Subdomain_Count"]
    ]
    for row_idx, row_data in enumerate(data):
        for col_idx, cell_data in enumerate(row_data):
            table.cell(row_idx + 1, col_idx).text = cell_data

    # Slide 6: Robustness
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Adversarial Robustness & Defense"
    
    sp = slide.shapes[1]
    sp.element.getparent().remove(sp.element)
    
    rows = 3
    cols = 4
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    
    headers = ["Model State", "Clean Data", "FGSM Attack", "PGD Attack"]
    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    data = [
        ["Standard LightGBM", "98.86%", "63.20%", "46.15%"],
        ["Adversarially Trained", "98.10%", "92.30%", "89.50%"]
    ]
    for row_idx, row_data in enumerate(data):
        for col_idx, cell_data in enumerate(row_data):
            table.cell(row_idx + 1, col_idx).text = cell_data

    # Slide 7: Architecture
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Hybrid Detection Engine & API"
    tf = slide.placeholders[1].text_frame
    tf.text = "Funnel Architecture for Real-Time Processing:"
    p = tf.add_paragraph()
    p.text = "1. Whitelist (O(1) instantaneous safe check)"
    p.level = 1
    p = tf.add_paragraph()
    p.text = "2. Blacklist (Immediate block of known threats)"
    p.level = 1
    p = tf.add_paragraph()
    p.text = "3. Heuristics Engine (Fast Regex rule validation)"
    p.level = 1
    p = tf.add_paragraph()
    p.text = "4. Machine Learning (LightGBM Zero-Day detection fallback)"
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Backend Deployment: FastAPI handling asynchronous inferences."
    p.level = 0
    p = tf.add_paragraph()
    p.text = "Frontend: Modern React Dashboard to visualize XAI explanations."
    p.level = 0

    # Slide 8: Conclusion
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Statistical Proof & Conclusion"
    tf = slide.placeholders[1].text_frame
    tf.text = "Statistical Validation:"
    p = tf.add_paragraph()
    p.text = "ANOVA p-value = 0.1133 confirms stable behavior across ensemble models."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Summary of Contributions:"
    p.level = 0
    p = tf.add_paragraph()
    p.text = "Built a high-performance detection system (98.86% Accuracy)."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Solved the Black Box problem using Triangular Validation (SHAP/LIME)."
    p.level = 1
    p = tf.add_paragraph()
    p.text = "Hardened the architecture using Adversarial Training against evasion."
    p.level = 1

    prs.save('Presentation_Phishing_PFE.pptx')
    print("Presentation saved successfully.")

if __name__ == '__main__':
    create_presentation()
