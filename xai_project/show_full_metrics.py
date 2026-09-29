"""
show_full_metrics.py — Tableau Complet des 4 Métriques
=======================================================
Affiche Accuracy, Precision, Recall, F1-Score et AUC-ROC
pour XGBoost, LightGBM et Random Forest.

Répond à la question de la superviseure :
  "Le travail se base-t-il seulement sur l'Accuracy ?"
  → NON : toutes les métriques sont calculées ici.

Usage:
    cd xai_project
    python show_full_metrics.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from utils.data_loader import load_phishing_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR   = ROOT / "outputs" / "models"
OUTPUTS_DIR  = ROOT / "outputs" / "validation"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_models():
    logger.info("Chargement des modeles...")
    models = {
        "XGBoost":      joblib.load(MODELS_DIR / "xgboost_model.pkl"),
        "LightGBM":     joblib.load(MODELS_DIR / "lightgbm_model.pkl"),
        "RandomForest": joblib.load(MODELS_DIR / "random_forest_model.pkl"),
    }
    logger.info(f"  {len(models)} modeles charges : {list(models.keys())}")
    return models


def load_data():
    logger.info("Chargement des donnees...")
    possible_paths = [
        ROOT.parent / "Codes" / "PhishFOE_ds" / "PhishFOE_ds.csv",
        ROOT.parent / "Codes" / "Phishing_Legitimate_full_codes" / "dataset_full.csv",
        ROOT / "data" / "dataset.csv",
    ]
    dataset_path = None
    for p in possible_paths:
        if p.exists():
            dataset_path = p
            break
    if dataset_path is None:
        raise FileNotFoundError(
            "Dataset introuvable. Chemins testes:\n"
            + "\n".join(str(p) for p in possible_paths)
        )
    logger.info(f"  Dataset : {dataset_path}")
    df_raw, X, y, feature_names = load_phishing_dataset(str(dataset_path))
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler  = joblib.load(scaler_path)
        X_test  = scaler.transform(X_test)
        logger.info("  Scaler applique.")
    else:
        logger.warning("  scaler.joblib introuvable.")

    logger.info(f"  Test set : {X_test.shape} | Phishing : {int(y_test.sum())} | Legitime : {int((y_test==0).sum())}")
    return X_test, y_test


def compute_all_metrics(model, X_test, y_test, model_name):
    """Calcule les 5 métriques pour un modèle."""
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model":     model_name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_test, y_pred_proba),
        "cm":        confusion_matrix(y_test, y_pred).tolist(),
        "report":    classification_report(y_test, y_pred,
                         target_names=["Legitimate", "Phishing"],
                         digits=4),
    }


def print_metrics_table(all_metrics):
    """Affiche un tableau bien formaté dans le terminal."""
    sep = "=" * 85
    thin = "-" * 85

    print("\n" + sep)
    print("  TABLEAU COMPLET DES METRIQUES — PhishShield (XAI Phishing Detection)")
    print("  Reponse a la superviseure : le systeme utilise 5 metriques, pas juste l'Accuracy")
    print(sep)

    header = f"  {'Modele':<16} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10}"
    print(header)
    print(thin)

    best_f1    = max(m["f1"]       for m in all_metrics)
    best_acc   = max(m["accuracy"] for m in all_metrics)

    for m in all_metrics:
        acc_str  = f"{m['accuracy']*100:.2f}%"
        prec_str = f"{m['precision']*100:.2f}%"
        rec_str  = f"{m['recall']*100:.2f}%"
        f1_str   = f"{m['f1']*100:.2f}%"
        auc_str  = f"{m['auc_roc']*100:.2f}%"

        # Marquer le meilleur modèle
        marker = " ★" if m["f1"] == best_f1 else "  "
        row = f"{marker} {m['model']:<14} {acc_str:>10} {prec_str:>10} {rec_str:>10} {f1_str:>10} {auc_str:>10}"
        print(row)

    print(thin)
    print("  ★ = Meilleur modele (F1-Score le plus eleve)")
    print(sep)

    # Explication des métriques pour la soutenance
    print("\n  EXPLICATION DES METRIQUES (pour ta superviseure) :")
    print(thin)
    print("  Accuracy  : % total de bonnes predictions (phishing + legitime)")
    print("  Precision : Parmi les URLs marquees PHISHING, combien sont vraiment du phishing ?")
    print("              → Faible = trop de Faux Positifs (vraies URLs bloquees par erreur)")
    print("  Recall    : Parmi toutes les vraies attaques, combien ont ete detectees ?")
    print("              → Faible = trop de Faux Negatifs (attaques non detectees = DANGER)")
    print("  F1-Score  : Equilibre parfait entre Precision et Recall (metrique principale)")
    print("  AUC-ROC   : Capacite du modele a distinguer les 2 classes (1.0 = parfait)")
    print(sep)


def print_confusion_matrices(all_metrics):
    """Affiche les matrices de confusion."""
    sep  = "=" * 85
    thin = "-" * 85
    print("\n" + sep)
    print("  MATRICES DE CONFUSION (Faux Positifs / Faux Negatifs)")
    print(sep)

    for m in all_metrics:
        cm = m["cm"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        print(f"\n  [ {m['model']} ]")
        print(thin)
        print(f"                    Predit : Legitime   Predit : Phishing")
        print(f"  Reel : Legitime        {tn:>6}           {fp:>6}  ← Faux Positifs (FP)")
        print(f"  Reel : Phishing        {fn:>6}           {tp:>6}")
        print(f"                                ↑")
        print(f"                      Faux Negatifs (FN) — URLs de phishing non detectees !")
        print(f"\n  → FP={fp} (URLs legitimes bloquees par erreur)")
        print(f"  → FN={fn} (attaques de phishing manquees = TRES DANGEREUX)")
    print("\n" + sep)


def save_metrics_json(all_metrics):
    """Sauvegarde les métriques en JSON."""
    output = {}
    for m in all_metrics:
        name = m["model"]
        output[name] = {
            "accuracy":  round(m["accuracy"],  6),
            "precision": round(m["precision"], 6),
            "recall":    round(m["recall"],    6),
            "f1":        round(m["f1"],        6),
            "auc_roc":   round(m["auc_roc"],   6),
            "confusion_matrix": m["cm"],
        }

    json_path = OUTPUTS_DIR / "full_metrics_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  [OK] Metriques sauvegardees : {json_path}")
    return json_path


def save_metrics_markdown(all_metrics):
    """Sauvegarde un rapport Markdown propre pour le rapport LaTeX."""
    lines = [
        "# Full Metrics Report — PhishShield",
        "",
        "## Performance Comparison (All Metrics)",
        "",
        "| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |",
        "|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        lines.append(
            f"| **{m['model']}** "
            f"| {m['accuracy']*100:.2f}% "
            f"| {m['precision']*100:.2f}% "
            f"| {m['recall']*100:.2f}% "
            f"| {m['f1']*100:.2f}% "
            f"| {m['auc_roc']*100:.2f}% |"
        )

    lines += [
        "",
        "## Confusion Matrices",
        "",
    ]
    for m in all_metrics:
        cm = m["cm"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        lines += [
            f"### {m['model']}",
            "",
            f"| | Predicted Legitimate | Predicted Phishing |",
            f"|---|---|---|",
            f"| **Actual Legitimate** | TN = {tn} | FP = {fp} |",
            f"| **Actual Phishing**   | FN = {fn} | TP = {tp} |",
            "",
            f"- **False Positives (FP) = {fp}** : Legitimate URLs incorrectly flagged as phishing",
            f"- **False Negatives (FN) = {fn}** : Phishing URLs that escaped detection ← Critical!",
            "",
        ]

    lines += [
        "## Classification Reports",
        "",
    ]
    for m in all_metrics:
        lines += [
            f"### {m['model']}",
            "```",
            m["report"],
            "```",
            "",
        ]

    md_path = OUTPUTS_DIR / "full_metrics_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  [OK] Rapport Markdown sauvegarde : {md_path}")
    return md_path


# ── POINT D'ENTREE ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Fix encodage Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n  PhishShield — Calcul de TOUTES les metriques")
    print("  =" * 42)

    try:
        models        = load_models()
        X_test, y_test = load_data()

        # Convertir en numpy si nécessaire
        import pandas as pd
        if hasattr(X_test,  "values"): X_test  = X_test.values
        if hasattr(y_test,  "values"): y_test  = y_test.values

        # Calculer les métriques pour chaque modèle
        all_metrics = []
        for name, model in models.items():
            logger.info(f"  Calcul des metriques pour {name}...")
            metrics = compute_all_metrics(model, X_test, y_test, name)
            all_metrics.append(metrics)

        # Afficher les résultats
        print_metrics_table(all_metrics)
        print_confusion_matrices(all_metrics)

        # Sauvegarder
        json_path = save_metrics_json(all_metrics)
        md_path   = save_metrics_markdown(all_metrics)

        print(f"\n  Fichiers generes :")
        print(f"    → {json_path}")
        print(f"    → {md_path}")
        print("\n  [TERMINE] Tu peux maintenant presenter les 5 metriques a ta superviseure !")

    except FileNotFoundError as e:
        logger.error(f"Fichier introuvable : {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur : {e}", exc_info=True)
        sys.exit(1)
