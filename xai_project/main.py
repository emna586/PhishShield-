"""
Script principal - Orchestration complete de la pipeline XAI + Robustesse Adversariale
Projet de detection de phishing - Licence L3
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Forcer l'encodage UTF-8 pour les logs sur Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import des modules
from config import (
    PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR, MODELS_DIR, VISUALIZATIONS_DIR,
    EXPLANATIONS_DIR, XAI_CONFIG, MODELS, CLASS_NAMES
)
from utils import setup_logger, data_loader, model_trainer
from xai import shap_explainer, lime_explainer, pfi_explainer, xai_validation
from utils.visualizer import (
    plot_feature_importance_comparison, plot_coherence_heatmap,
    plot_overlap_analysis, save_figure
)
from report_generator import generate_xai_report, save_report, add_robustness_section
from robustness.robustness_eval import evaluate_robustness, save_metrics, plot_robustness_comparison

logger = setup_logger(__name__)


def main():
    """
    Pipeline principale: Chargement -> Entraînement -> XAI -> Robustesse -> Rapport -> API
    """

    logger.info("=" * 70)
    logger.info("   XAI PHISHING DETECTION PIPELINE")
    logger.info("=" * 70)

    # ============================================================================
    # ETAPE 1: Chargement et Preparation des Donnees
    # ============================================================================

    logger.info("\n[ETAPE 1] CHARGEMENT ET PREPARATION DES DONNEES\n")

    # Chercher le fichier CSV dans les emplacements possibles
    csv_path = None
    possible_paths = [
        DATA_DIR / "phishing.csv",
        Path(PROJECT_ROOT).parent / "Codes" / "PhishFOE_ds" / "PhishFOE_ds.csv",
        Path("data/phishing.csv"),
        Path("../Codes/PhishFOE_ds/PhishFOE_ds.csv"),
    ]

    for possible_path in possible_paths:
        if Path(possible_path).exists():
            csv_path = possible_path
            break

    if csv_path is None:
        logger.error("[ERREUR] Dataset CSV non trouve!")
        logger.info("Chemins verifies:")
        for p in possible_paths:
            logger.info(f"  - {p}")
        return False

    logger.info(f"[OK] Dataset trouve: {csv_path}")

    # Preparer les donnees
    data_dict = data_loader.prepare_data(str(csv_path))

    X_train       = data_dict['X_train']
    X_test        = data_dict['X_test']
    y_train       = data_dict['y_train']
    y_test        = data_dict['y_test']
    feature_names = data_dict['feature_names']
    scaler        = data_dict['scaler']

    # Sauvegarder le scaler pour l'API
    import joblib
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"[OK] Scaler sauvegarde: {scaler_path}")

    # ============================================================================
    # ETAPE 2: Entrainement des Modeles
    # ============================================================================

    logger.info("\n[ETAPE 2] ENTRAINEMENT DES MODELES\n")

    models  = model_trainer.train_all_models(X_train, y_train)
    metrics = model_trainer.evaluate_all_models(models, X_test, y_test)
    model_trainer.save_models(models)

    logger.info("[OK] Tous les modeles entraines et sauvegardes.")

    # ============================================================================
    # ETAPE 3: Calcul XAI (SHAP + LIME + PFI) pour Chaque Modele
    # ============================================================================

    logger.info("\n[ETAPE 3] CALCUL XAI POUR CHAQUE MODELE\n")

    xai_results = {}

    for model_key, model_obj in models.items():
        model_name = MODELS[model_key]['name']
        logger.info(f"\n{'─' * 70}")
        logger.info(f"Calcul XAI pour: {model_name}")
        logger.info(f"{'─' * 70}")

        # -- SHAP --
        shap_importance, top10_shap, shap_values_full = shap_explainer.calculate_shap(
            model_obj, X_test, feature_names, model_name,
            sample_size=XAI_CONFIG['shap'].get('sample_size')
        )

        # -- LIME --
        lime_importance, top10_lime = lime_explainer.calculate_lime(
            model_obj, X_train, X_test, feature_names, model_name,
            num_samples=XAI_CONFIG['lime'].get('num_samples', 10),
            num_features=XAI_CONFIG['lime'].get('num_features_lime')
        )

        # -- PFI --
        pfi_importance, top10_pfi = pfi_explainer.calculate_pfi(
            model_obj, X_test, y_test, feature_names, model_name,
            n_repeats=XAI_CONFIG['pfi'].get('n_repeats', 10)
        )

        # -- Validation Croisee Triangulaire --
        validation_result = xai_validation.triangular_xai_validation(
            shap_importance, lime_importance, pfi_importance,
            top10_shap, top10_lime, top10_pfi,
            feature_names, model_name
        )

        xai_results[model_key] = {
            'importances': {
                'shap': shap_importance,
                'lime': lime_importance,
                'pfi':  pfi_importance
            },
            'top_indices': {
                'shap': top10_shap,
                'lime': top10_lime,
                'pfi':  top10_pfi
            },
            'validation': validation_result
        }

    # ============================================================================
    # ETAPE 4: Generation des Visualisations
    # ============================================================================

    logger.info("\n[ETAPE 4] GENERATION DES VISUALISATIONS\n")

    for model_key, xai_data in xai_results.items():
        model_name      = MODELS[model_key]['name']
        importances_dict = xai_data['importances']
        validation_result = xai_data['validation']

        logger.info(f"Visualisations pour: {model_name}")

        # Comparaison des 3 methodes
        fig, axes = plot_feature_importance_comparison(importances_dict, feature_names, top_n=15)
        save_figure(fig, f"xai_comparison_{model_key}")

        # Heatmap de coherence
        fig, ax = plot_coherence_heatmap(validation_result['correlations'], model_name)
        save_figure(fig, f"coherence_heatmap_{model_key}")

        # Analyse d'overlap
        fig, ax = plot_overlap_analysis(validation_result['overlaps'], model_name)
        save_figure(fig, f"overlap_analysis_{model_key}")

    # ============================================================================
    # ETAPE 5: Generation du Rapport XAI Markdown
    # ============================================================================

    logger.info("\n[ETAPE 5] GENERATION DU RAPPORT XAI MARKDOWN\n")

    results_for_report = {
        model_key: xai_data['validation']
        for model_key, xai_data in xai_results.items()
    }

    report_content = generate_xai_report(results_for_report, feature_names, metrics)
    report_path    = EXPLANATIONS_DIR / "xai_report.md"
    save_report(report_content, report_path)

    # ============================================================================
    # ETAPE 6: Evaluation de Robustesse Adversariale (FGSM + PGD)
    # ============================================================================

    logger.info("\n[ETAPE 6] ROBUSTESSE ADVERSARIALE — FGSM + PGD\n")

    robustness_results = evaluate_robustness(
        models       = models,
        X_test       = X_test,
        y_test       = y_test,
        feature_names = feature_names,
        fgsm_epsilon  = 0.1,
        pgd_epsilon   = 0.1,
        pgd_steps     = 40,
        pgd_alpha     = 0.01,
        fgsm_sample   = 200,
        pgd_sample    = 100,
    )

    # Sauvegarder metriques JSON
    save_metrics(robustness_results)

    # Generer visualisations
    plot_robustness_comparison(robustness_results)

    # Ajouter section robustesse au rapport Markdown
    robustness_section = add_robustness_section(robustness_results)
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(robustness_section)
    logger.info(f"[OK] Section robustesse ajoutee au rapport: {report_path}")

    # ============================================================================
    # RESUME FINAL
    # ============================================================================

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE AVEC SUCCES!")
    logger.info("=" * 70)

    logger.info("\nSommaire XAI:")
    for model_key, xai_data in xai_results.items():
        model_name    = MODELS[model_key]['name']
        validation    = xai_data['validation']
        consensus_count = len(validation['consensus_features_names'])
        coherent_str  = "OK" if validation['coherent'] else "INCOHERENT"
        logger.info(f"\n  {model_name}:")
        logger.info(f"    - Coherence          : {coherent_str}")
        logger.info(f"    - Features Consensus : {consensus_count}/10")
        logger.info(f"    - Verifications      : {validation['passed']}/{validation['total']} passees")

    logger.info("\nSommaire Robustesse Adversariale:")
    for model_key, rob_data in robustness_results.items():
        logger.info(f"\n  {rob_data['model_name']}:")
        logger.info(f"    - Baseline      : {rob_data['baseline_accuracy']:.4f}")
        logger.info(f"    - FGSM Accuracy : {rob_data['fgsm_accuracy']:.4f}  (drop: {rob_data['fgsm_drop']:.4f})")
        logger.info(f"    - PGD Accuracy  : {rob_data['pgd_accuracy']:.4f}  (drop: {rob_data['pgd_drop']:.4f})")
        logger.info(f"    - Niveau        : {rob_data['robustness_level']}")

    logger.info("\n" + "=" * 70)
    logger.info("Rapport XAI      : explanations/xai_report.md")
    logger.info("Metriques robu   : outputs/robustness/metrics.json")
    logger.info("Modeles sauves   : outputs/models/")
    logger.info("Visualisations   : outputs/visualizations/")
    logger.info("=" * 70)
    logger.info("\nPour lancer l'API FastAPI:")
    logger.info("  uvicorn api.main:app --reload --port 8000")
    logger.info("  Puis ouvrir: http://localhost:8000")
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"[ERREUR FATALE] {e}", exc_info=True)
        sys.exit(1)
