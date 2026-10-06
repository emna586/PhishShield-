<div align="center">
  <h1>🛡️ PhishShield</h1>
  <p><strong>An Intelligent, Hybrid & Explainable Phishing Detection System</strong></p>
  <p><i>Achieving 98.88% Accuracy with Structural Adversarial Immunity</i></p>
</div>

---

## 📖 About The Project
**PhishShield** is a comprehensive, end-to-end cybersecurity solution developed to detect phishing URLs with extremely high accuracy while completely solving the "black box" problem of traditional AI models. 

This project explores the intersection of **Machine Learning, Deep Learning, Explainable AI (XAI), and Adversarial Robustness**. By comparing complex Neural Networks (CNN, LSTM, MLP) against tree-based ensembles (LightGBM, XGBoost, Random Forest), this research empirically proves that highly-optimized tree models not only outperform Deep Learning in tabular cybersecurity data, but also provide inherent mathematical immunity against gradient-based adversarial attacks.

## 🚀 Key Features & Methodologies

* **Advanced Feature Engineering:** Extraction of 29 highly discriminative URL-based and HTML-based features from the balanced 100K+ *PhishFOE* dataset.
* **Extensive Model Benchmarking:** Head-to-head comparison between state-of-the-art Deep Learning models (LSTM, CNN, MLP) and Machine Learning ensembles. **LightGBM** emerged as the champion (98.88% F1-Score).
* **Explainable AI (XAI):** Integration of **SHAP** (Shapley Additive exPlanations) and **LIME** to provide global feature importance and local, human-readable explanations for every single prediction, making the AI auditable for Security Operations Centers (SOCs).
* **Adversarial Robustness Testing:** Rigorous stress-testing of models using white-box **FGSM** and **PGD** attacks. The project empirically demonstrates that while LSTM/CNN models collapse under PGD attacks (dropping to ~54% accuracy), LightGBM maintains >98% accuracy due to its non-differentiable decision boundaries.
* **Full-Stack Production Deployment:** The robust LightGBM model is deployed via a high-performance **FastAPI** backend, serving predictions in real-time to a modern **Next.js** dashboard.

---

## 🛠️ Skills & Technologies Demonstrated

This project serves as a comprehensive portfolio piece, demonstrating a strong command over the entire Data Science and Software Engineering lifecycle:

### 🧠 Artificial Intelligence & Data Science
* **Machine Learning:** LightGBM, XGBoost, Random Forest, Feature Selection, Hyperparameter Tuning (Optuna/GridSearch).
* **Deep Learning:** TensorFlow, Keras, CNNs (Conv1D), Recurrent Neural Networks (LSTM), Multilayer Perceptrons.
* **Explainable AI (XAI):** Deep understanding and implementation of SHAP and LIME for model interpretability.
* **Data Processing:** Pandas, NumPy, Scikit-Learn, rigorous stratified splitting, and data balancing.

### 🛡️ Cybersecurity
* **Threat Intelligence:** In-depth understanding of phishing vectors, URL manipulation, and HTML evasion techniques.
* **Adversarial Machine Learning:** Formulating and executing white-box gradient attacks (Fast Gradient Sign Method, Projected Gradient Descent) to evaluate model vulnerabilities.

### 💻 Software Engineering & Deployment
* **Backend Development:** Python, **FastAPI** (RESTful API design, request validation, low-latency inference).
* **Frontend Development:** JavaScript/TypeScript, **React**, **Next.js** (Building interactive dashboards and data visualizations).
* **Version Control & Best Practices:** Git, GitHub, modular code architecture, and robust documentation.

---

## 📂 Repository Structure
* `/Codes/` - Jupyter Notebooks containing the model training, cross-validation, and benchmarking.
* `/xai_project/` - The core Python application containing the ML pipeline, XAI explainers, adversarial robustness testing, and the FastAPI backend.
* `/phishshield-frontend/` - The Next.js web application for the real-time detection dashboard.
* `/Reports/` - Detailed research reports, PFE documentation, and presentation materials.

---
*Developed as a Final Year Engineering Project (PFE) by Emna.*
