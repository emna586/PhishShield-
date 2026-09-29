"""
train_deep_learning.py — Entraînement CNN, LSTM, MLP
=====================================================
Entraîne 3 modèles Deep Learning sur le dataset PhishFOE
et calcule Accuracy, Precision, Recall, F1-Score, AUC-ROC.

Répond à la demande de la superviseure :
  "Ajoutez CNN, LSTM et MLP avec les 4 métriques."

Usage:
    cd xai_project
    python train_deep_learning.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

# Fix UTF-8 Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR  = ROOT / "outputs" / "models"
OUTPUTS_DIR = ROOT / "outputs" / "validation"
DL_DIR      = ROOT / "outputs" / "models" / "deep_learning"
DL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Chargement données ─────────────────────────────────────────────────────
def load_data():
    from utils.data_loader import load_phishing_dataset
    from sklearn.model_selection import train_test_split

    possible_paths = [
        ROOT.parent / "Codes" / "PhishFOE_ds" / "PhishFOE_ds.csv",
        ROOT.parent / "Codes" / "Phishing_Legitimate_full_codes" / "Phishing_Legitimate_full.csv",
    ]
    dataset_path = next((p for p in possible_paths if p.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Dataset introuvable.")

    logger.info(f"Dataset: {dataset_path}")
    df_raw, X, y, feature_names = load_phishing_dataset(str(dataset_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler  = joblib.load(scaler_path)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)
        logger.info("Scaler applique.")

    # Convertir en numpy
    if hasattr(X_train, "values"): X_train = X_train.values
    if hasattr(X_test,  "values"): X_test  = X_test.values
    if hasattr(y_train, "values"): y_train = y_train.values
    if hasattr(y_test,  "values"): y_test  = y_test.values

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names


# ── 2. Métriques ──────────────────────────────────────────────────────────────
def compute_metrics(y_test, y_pred, y_proba, model_name):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0)    * 100, 2),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0)        * 100, 2),
        "auc_roc":   round(roc_auc_score(y_test, y_proba)  * 100, 2),
        "fp": int(fp), "fn": int(fn),
        "tp": int(tp), "tn": int(tn),
    }


# ── 3. Modèles Deep Learning ──────────────────────────────────────────────────
def build_mlp(input_dim):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ], name="MLP")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn(input_dim):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim, 1)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ], name="CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_lstm(input_dim):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim, 1)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ], name="LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── 4. Entraînement ───────────────────────────────────────────────────────────
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, reshape=False):
    import tensorflow as tf

    X_tr = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) if reshape else X_train
    X_te = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)    if reshape else X_test

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ]

    logger.info(f"\n{'='*50}")
    logger.info(f"Entrainement : {model_name}")
    logger.info(f"{'='*50}")

    history = model.fit(
        X_tr, y_train,
        epochs=50,
        batch_size=512,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # Prédictions
    y_proba = model.predict(X_te, batch_size=512, verbose=0).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)

    # Métriques
    metrics = compute_metrics(y_test, y_pred, y_proba, model_name)

    # Sauvegarder le modèle
    save_path = DL_DIR / f"{model_name.lower()}_model.keras"
    model.save(save_path)
    logger.info(f"Modele sauvegarde: {save_path}")

    return metrics, history


# ── 5. Affichage du tableau comparatif ───────────────────────────────────────
def print_comparison_table(dl_metrics, ml_metrics_path):
    """Affiche ML + DL côte à côte."""
    sep  = "=" * 90
    thin = "-" * 90

    # Charger métriques ML existantes
    ml_data = []
    if ml_metrics_path.exists():
        with open(ml_metrics_path, "r", encoding="utf-8") as f:
            ml_json = json.load(f)
        for name, m in ml_json.items():
            ml_data.append({
                "model":     name + " (ML)",
                "accuracy":  round(m["accuracy"]  * 100, 2),
                "precision": round(m["precision"] * 100, 2),
                "recall":    round(m["recall"]    * 100, 2),
                "f1":        round(m["f1"]        * 100, 2),
                "auc_roc":   round(m["auc_roc"]   * 100, 2),
                "fp": m["confusion_matrix"][0][1],
                "fn": m["confusion_matrix"][1][0],
            })

    all_metrics = ml_data + dl_metrics
    all_metrics.sort(key=lambda x: x["f1"], reverse=True)

    print("\n" + sep)
    print("  TABLEAU COMPARATIF COMPLET : Machine Learning vs Deep Learning")
    print(sep)
    print(f"  {'Modele':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC-ROC':>9} {'FP':>5} {'FN':>5}")
    print(thin)

    for i, m in enumerate(all_metrics):
        marker = " ★" if i == 0 else "  "
        tag    = "(DL)" if "(ML)" not in m["model"] else ""
        print(
            f"{marker} {m['model']:<20} {tag:<5}"
            f"{m['accuracy']:>8.2f}%"
            f"{m['precision']:>9.2f}%"
            f"{m['recall']:>8.2f}%"
            f"{m['f1']:>7.2f}%"
            f"{m['auc_roc']:>8.2f}%"
            f"{m['fp']:>6}"
            f"{m['fn']:>6}"
        )

    print(thin)
    print("  ★ = Meilleur modele global (F1-Score)")
    print("  FP = Faux Positifs | FN = Faux Negatifs (attaques manquees)")
    print(sep)
    return all_metrics


# ── 6. Sauvegarder résultats ──────────────────────────────────────────────────
def save_all_results(all_metrics):
    out = {}
    for m in all_metrics:
        out[m["model"]] = {k: v for k, v in m.items() if k != "model"}

    json_path = OUTPUTS_DIR / "full_comparison_ml_vs_dl.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"[OK] Resultats sauvegardes: {json_path}")

    # Markdown pour le rapport LaTeX
    md_lines = [
        "# ML vs Deep Learning — Full Comparison",
        "",
        "| Model | Type | Accuracy | Precision | Recall | F1-Score | AUC-ROC | FP | FN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        mtype = "ML" if "(ML)" in m["model"] else "**DL**"
        name  = m["model"].replace(" (ML)", "").replace(" (DL)", "")
        md_lines.append(
            f"| **{name}** | {mtype} "
            f"| {m['accuracy']:.2f}% "
            f"| {m['precision']:.2f}% "
            f"| {m['recall']:.2f}% "
            f"| {m['f1']:.2f}% "
            f"| {m['auc_roc']:.2f}% "
            f"| {m['fp']} | {m['fn']} |"
        )
    md_lines += [
        "",
        "> FP = False Positives (legitimate URLs incorrectly blocked)",
        "> FN = False Negatives (phishing attacks missed — CRITICAL)",
    ]
    md_path = OUTPUTS_DIR / "full_comparison_ml_vs_dl.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"[OK] Rapport Markdown: {md_path}")
    return json_path, md_path


# ── POINT D'ENTREE ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        # Limiter les logs TensorFlow
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        logger.error("TensorFlow n'est pas installe !")
        logger.info("Installez-le avec : pip install tensorflow")
        sys.exit(1)

    logger.info("PhishShield — Entrainement Deep Learning (CNN + LSTM + MLP)")
    logger.info("=" * 60)

    # 1. Charger données
    X_train, X_test, y_train, y_test, feature_names = load_data()
    input_dim = X_train.shape[1]
    logger.info(f"Dimension des features : {input_dim}")

    dl_metrics = []

    # 2. MLP
    mlp     = build_mlp(input_dim)
    m_mlp, _ = train_and_evaluate(mlp, X_train, X_test, y_train, y_test, "MLP", reshape=False)
    dl_metrics.append(m_mlp)
    print(f"\n  MLP   → Accuracy={m_mlp['accuracy']}% | F1={m_mlp['f1']}% | AUC={m_mlp['auc_roc']}%")

    # 3. CNN
    cnn     = build_cnn(input_dim)
    m_cnn, _ = train_and_evaluate(cnn, X_train, X_test, y_train, y_test, "CNN", reshape=True)
    dl_metrics.append(m_cnn)
    print(f"\n  CNN   → Accuracy={m_cnn['accuracy']}% | F1={m_cnn['f1']}% | AUC={m_cnn['auc_roc']}%")

    # 4. LSTM
    lstm    = build_lstm(input_dim)
    m_lstm, _ = train_and_evaluate(lstm, X_train, X_test, y_train, y_test, "LSTM", reshape=True)
    dl_metrics.append(m_lstm)
    print(f"\n  LSTM  → Accuracy={m_lstm['accuracy']}% | F1={m_lstm['f1']}% | AUC={m_lstm['auc_roc']}%")

    # 5. Tableau comparatif
    ml_metrics_path = OUTPUTS_DIR / "full_metrics_report.json"
    all_metrics     = print_comparison_table(dl_metrics, ml_metrics_path)

    # 6. Sauvegarder
    json_path, md_path = save_all_results(all_metrics)

    print(f"\n  Fichiers generes :")
    print(f"    → {json_path}")
    print(f"    → {md_path}")
    print(f"    → {DL_DIR}/mlp_model.keras")
    print(f"    → {DL_DIR}/cnn_model.keras")
    print(f"    → {DL_DIR}/lstm_model.keras")
    print("\n  [TERMINE] Comparaison ML vs DL complete !")
