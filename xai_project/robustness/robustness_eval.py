"""
Robustness Evaluation — Évaluation complète FGSM + PGD
========================================================
Orchestre l'évaluation de robustesse pour tous les modèles,
génère les métriques JSON et les visualisations.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

from config import OUTPUTS_DIR
from utils import setup_logger
from .fgsm_attack import fgsm_attack
from .pgd_attack import pgd_attack
from .adversarial_examples import (
    compute_perturbation_stats,
    plot_perturbation_by_feature,
    save_adversarial_examples,
)

logger = setup_logger(__name__)

# ── Répertoires de sortie ────────────────────────────────────────────────────
ROBUSTNESS_DIR   = OUTPUTS_DIR / "robustness"
ADV_EXAMPLES_DIR = ROBUSTNESS_DIR / "adversarial_examples"
ROB_VIZ_DIR      = ROBUSTNESS_DIR / "visualizations"

for d in [ROBUSTNESS_DIR, ADV_EXAMPLES_DIR, ROB_VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ÉVALUATION PRINCIPALE
# ============================================================================

def evaluate_robustness(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    fgsm_epsilon: float = 0.1,
    pgd_epsilon: float  = 0.1,
    pgd_steps: int      = 40,
    pgd_alpha: float    = 0.01,
    fgsm_sample: int    = 200,
    pgd_sample: int     = 100,
) -> Dict[str, Any]:
    """
    Évaluation complète de robustesse adversariale pour tous les modèles.

    Étapes pour chaque modèle :
        1. Accuracy baseline (données propres)
        2. Attaque FGSM — une seule étape
        3. Attaque PGD  — itérative (la plus forte)
        4. Calcul des métriques (drops, robustness score)
        5. Stats de perturbation par feature

    Args:
        models        : Dict {'xgboost': model, 'lightgbm': model, ...}
        X_test        : Données de test propres
        y_test        : Vraies étiquettes
        feature_names : Noms des features
        fgsm_epsilon  : Force perturbation FGSM
        pgd_epsilon   : Rayon boule PGD
        pgd_steps     : Nombre d'itérations PGD
        pgd_alpha     : Taille de pas PGD
        fgsm_sample   : Nombre de samples pour FGSM
        pgd_sample    : Nombre de samples pour PGD (plus petit car plus lent)

    Returns:
        results : Dict complet des métriques par modèle
    """
    results = {}

    logger.info("=" * 70)
    logger.info("   EVALUATION DE ROBUSTESSE ADVERSARIALE")
    logger.info("=" * 70)

    # Noms lisibles pour les logs et le rapport
    model_display_names = {
        'xgboost'      : 'XGBoost',
        'lightgbm'     : 'LightGBM',
        'random_forest': 'Random Forest',
    }

    for model_key, model_obj in models.items():
        model_name = model_display_names.get(model_key, model_key)

        logger.info(f"\n{'─' * 70}")
        logger.info(f"  Modèle : {model_name}")
        logger.info(f"{'─' * 70}")

        # ── 1. Accuracy Baseline ─────────────────────────────────────────────
        baseline_acc = float(model_obj.score(X_test, y_test))
        logger.info(f"  [OK] Accuracy Baseline : {baseline_acc:.4f}  ({baseline_acc*100:.2f}%)")

        # ── 2. Attaque FGSM ──────────────────────────────────────────────────
        logger.info(f"\n  [FGSM] epsilon={fgsm_epsilon} | samples={fgsm_sample}")
        fgsm_result = fgsm_attack(
            model_obj, X_test, y_test,
            epsilon=fgsm_epsilon,
            sample_size=fgsm_sample,
        )
        fgsm_acc  = fgsm_result['accuracy']
        fgsm_drop = baseline_acc - fgsm_acc
        logger.info(f"  [FGSM] Accuracy : {fgsm_acc:.4f}  ({fgsm_acc*100:.2f}%)")
        logger.info(f"  [FGSM] Drop     : {fgsm_drop:.4f}  ({fgsm_drop*100:.2f}%)")

        # ── 3. Attaque PGD ───────────────────────────────────────────────────
        logger.info(f"\n  [PGD] epsilon={pgd_epsilon} | steps={pgd_steps} | alpha={pgd_alpha} | samples={pgd_sample}")
        pgd_result = pgd_attack(
            model_obj, X_test, y_test,
            epsilon=pgd_epsilon,
            steps=pgd_steps,
            alpha=pgd_alpha,
            sample_size=pgd_sample,
        )
        pgd_acc  = pgd_result['accuracy']
        pgd_drop = baseline_acc - pgd_acc
        logger.info(f"  [PGD]  Accuracy : {pgd_acc:.4f}  ({pgd_acc*100:.2f}%)")
        logger.info(f"  [PGD]  Drop     : {pgd_drop:.4f}  ({pgd_drop*100:.2f}%)")

        # ── 4. Robustness Score ──────────────────────────────────────────────
        # Moyenne pondérée : on donne plus de poids à PGD (plus forte)
        robustness_score = (baseline_acc + fgsm_acc + pgd_acc) / 3
        logger.info(f"\n  [SCORE] Robustness Score : {robustness_score:.4f}  ({robustness_score*100:.2f}%)")

        # Evaluation qualitative (ASCII uniquement pour compatibilite Windows)
        if pgd_drop <= 0.05:
            robustness_level = "[VERT] ROBUSTE"
        elif pgd_drop <= 0.15:
            robustness_level = "[JAUNE] MODERE"
        elif pgd_drop <= 0.30:
            robustness_level = "[ORANGE] VULNERABLE"
        else:
            robustness_level = "[ROUGE] TRES VULNERABLE"
        logger.info(f"  Niveau de robustesse : {robustness_level}")

        # ── 5. Stats de perturbation ─────────────────────────────────────────
        fgsm_pert_stats = compute_perturbation_stats(
            fgsm_result['X_clean'], fgsm_result['X_adv'], feature_names
        )
        pgd_pert_stats = compute_perturbation_stats(
            pgd_result['X_clean'], pgd_result['X_adv'], feature_names
        )

        # ── 6. Graphique perturbation par feature ─────────────────────────────
        # Aligner les arrays (FGSM et PGD ont potentiellement des tailles différentes)
        min_n = min(len(fgsm_result['X_clean']), len(pgd_result['X_clean']))
        plot_perturbation_by_feature(
            fgsm_result['X_clean'][:min_n],
            fgsm_result['X_adv'][:min_n],
            pgd_result['X_adv'][:min_n],
            feature_names, model_name, ROB_VIZ_DIR
        )

        # ── 7. Sauvegarder exemples adverses (FGSM) ──────────────────────────
        y_pred_clean_fgsm = model_obj.predict(fgsm_result['X_clean'])
        y_pred_adv_fgsm   = model_obj.predict(fgsm_result['X_adv'])
        save_adversarial_examples(
            fgsm_result['X_clean'], fgsm_result['X_adv'],
            fgsm_result['y_true'], y_pred_clean_fgsm, y_pred_adv_fgsm,
            feature_names, model_name, 'FGSM', ADV_EXAMPLES_DIR
        )

        # ── 8. Sauvegarder exemples adverses (PGD) ───────────────────────────
        y_pred_clean_pgd = model_obj.predict(pgd_result['X_clean'])
        y_pred_adv_pgd   = model_obj.predict(pgd_result['X_adv'])
        save_adversarial_examples(
            pgd_result['X_clean'], pgd_result['X_adv'],
            pgd_result['y_true'], y_pred_clean_pgd, y_pred_adv_pgd,
            feature_names, model_name, 'PGD', ADV_EXAMPLES_DIR
        )

        # ── 9. Stocker tous les résultats ─────────────────────────────────────
        results[model_key] = {
            'model_name'      : model_name,
            'baseline_accuracy': baseline_acc,
            'fgsm_accuracy'   : fgsm_acc,
            'pgd_accuracy'    : pgd_acc,
            'fgsm_drop'       : fgsm_drop,
            'pgd_drop'        : pgd_drop,
            'robustness_score': robustness_score,
            'robustness_level': robustness_level,
            'fgsm_perturbation': fgsm_pert_stats['global'],
            'pgd_perturbation' : pgd_pert_stats['global'],
            'fgsm_top_perturbed': list(fgsm_pert_stats['per_feature'].keys())[:10],
            'pgd_top_perturbed' : list(pgd_pert_stats['per_feature'].keys())[:10],
            'pgd_accuracy_history': pgd_result.get('accuracy_history', []),
            # Arrays bruts (non inclus dans metrics.json)
            '_X_adv_fgsm'     : fgsm_result['X_adv'],
            '_X_adv_pgd'      : pgd_result['X_adv'],
        }

    return results


# ============================================================================
# SAUVEGARDE MÉTRIQUES
# ============================================================================

def save_metrics(results: Dict[str, Any], output_path: Path = None) -> None:
    """
    Sauvegarde les métriques de robustesse dans un fichier metrics.json.
    Exclut les arrays numpy (trop volumineux).

    Args:
        results     : Dict des résultats de evaluate_robustness()
        output_path : Chemin de sauvegarde (défaut: outputs/robustness/metrics.json)
    """
    if output_path is None:
        output_path = ROBUSTNESS_DIR / "metrics.json"

    # Nettoyer les clés privées (arrays numpy)
    results_clean = {}
    for model_key, metrics in results.items():
        results_clean[model_key] = {
            k: v for k, v in metrics.items()
            if not k.startswith('_')
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Métriques robustesse sauvegardées: {output_path}")


# ============================================================================
# VISUALISATIONS GLOBALES
# ============================================================================

def plot_robustness_comparison(results: Dict[str, Any]) -> None:
    """
    Génère 3 graphiques de comparaison de robustesse entre tous les modèles :
        1. Bar plot : Accuracy (Baseline vs FGSM vs PGD) par modèle
        2. Bar plot : Drops d'accuracy (FGSM vs PGD) par modèle
        3. Heatmap : Matrice des drops

    Args:
        results : Dict des résultats de evaluate_robustness()
    """
    logger.info("Generation des visualisations de robustesse...")

    model_names    = [r['model_name']       for r in results.values()]
    baseline_accs  = [r['baseline_accuracy'] for r in results.values()]
    fgsm_accs      = [r['fgsm_accuracy']     for r in results.values()]
    pgd_accs       = [r['pgd_accuracy']      for r in results.values()]
    fgsm_drops     = [r['fgsm_drop']         for r in results.values()]
    pgd_drops      = [r['pgd_drop']          for r in results.values()]
    rob_scores     = [r['robustness_score']  for r in results.values()]

    # ── Graphique 1 : Accuracy Baseline vs FGSM vs PGD ──────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))

    x     = np.arange(len(model_names))
    width = 0.25

    b1 = ax.bar(x - width, baseline_accs, width, label='Baseline (propre)',
                color='#2ECC71', alpha=0.88)
    b2 = ax.bar(x,          fgsm_accs,    width, label='FGSM (ε=0.1)',
                color='#E67E22', alpha=0.88)
    b3 = ax.bar(x + width,  pgd_accs,     width, label='PGD (ε=0.1, 40 étapes)',
                color='#E74C3C', alpha=0.88)

    # Valeurs sur les barres
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Modèle', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('Robustesse Adversariale — Accuracy sous Attaque',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.12])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p1 = ROB_VIZ_DIR / "robustness_accuracy_comparison.png"
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  [OK] Graphique 1 sauvegarde: {p1}")

    # ── Graphique 2 : Drops d'accuracy ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))

    x     = np.arange(len(model_names))
    width = 0.35

    b1 = ax.bar(x - width / 2, [d * 100 for d in fgsm_drops], width,
                label='Drop FGSM (%)', color='#E67E22', alpha=0.88)
    b2 = ax.bar(x + width / 2, [d * 100 for d in pgd_drops],  width,
                label='Drop PGD (%)',  color='#E74C3C', alpha=0.88)

    # Ligne seuil : drop > 20% = vulnérable
    ax.axhline(y=20, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
               label='Seuil vulnérabilité (20%)')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Modèle', fontweight='bold', fontsize=12)
    ax.set_ylabel("Drop d'Accuracy (%)", fontweight='bold', fontsize=12)
    ax.set_title("Drop d'Accuracy sous Attaque FGSM vs PGD",
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim([0, max(max(pgd_drops), 0.1) * 130])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p2 = ROB_VIZ_DIR / "robustness_drops_comparison.png"
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  [OK] Graphique 2 sauvegarde: {p2}")

    # ── Graphique 3 : Heatmap des métriques ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Matrice : lignes = modèles, colonnes = métriques
    matrix = np.array([
        baseline_accs,
        fgsm_accs,
        pgd_accs,
        fgsm_drops,
        pgd_drops,
        rob_scores,
    ]).T  # shape (n_models, 6)

    col_labels = ['Baseline', 'FGSM Acc', 'PGD Acc',
                  'Drop FGSM', 'Drop PGD', 'Robustness Score']

    sns.heatmap(
        matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
        xticklabels=col_labels, yticklabels=model_names,
        ax=ax, linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Valeur'},
        vmin=0, vmax=1
    )

    ax.set_title('Matrice des Métriques de Robustesse',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    p3 = ROB_VIZ_DIR / "robustness_heatmap.png"
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  [OK] Graphique 3 sauvegarde: {p3}")

    # ── Graphique 4 : Évolution de l'accuracy PGD (convergence) ─────────────
    _has_history = any(
        len(r.get('pgd_accuracy_history', [])) > 0
        for r in results.values()
    )

    if _has_history:
        fig, ax = plt.subplots(figsize=(12, 6))

        colors_map = {
            'XGBoost'      : '#3498DB',
            'LightGBM'     : '#9B59B6',
            'Random Forest': '#2ECC71',
        }

        for model_key, result in results.items():
            history = result.get('pgd_accuracy_history', [])
            if history:
                steps_list = [h['step']     for h in history]
                accs_list  = [h['accuracy'] for h in history]
                ax.plot(steps_list, accs_list,
                        label=result['model_name'],
                        color=colors_map.get(result['model_name'], 'gray'),
                        linewidth=2.5, marker='o', markersize=5)

        ax.set_xlabel("Étape PGD", fontweight='bold', fontsize=12)
        ax.set_ylabel("Accuracy", fontweight='bold', fontsize=12)
        ax.set_title("Convergence de l'Attaque PGD (Accuracy par Étape)",
                     fontweight='bold', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        p4 = ROB_VIZ_DIR / "pgd_convergence.png"
        fig.savefig(p4, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  [OK] Graphique 4 (convergence PGD) sauvegarde: {p4}")

    logger.info("[OK] Toutes les visualisations de robustesse generees.")
