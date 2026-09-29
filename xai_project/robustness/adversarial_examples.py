"""
Adversarial Examples — Génération et visualisation des exemples adverses
=========================================================================
Fonctions utilitaires pour analyser et visualiser les exemples adverses
générés par FGSM et PGD.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif (sauvegarde fichier)
from pathlib import Path
from typing import List, Optional
from utils import setup_logger

logger = setup_logger(__name__)


def compute_perturbation_stats(X_clean: np.ndarray, X_adv: np.ndarray,
                                feature_names: List[str]) -> dict:
    """
    Calcule les statistiques de perturbation entre données propres et adverses.

    Args:
        X_clean       : Données propres (n_samples, n_features)
        X_adv         : Données adverses (n_samples, n_features)
        feature_names : Noms des features

    Returns:
        dict avec statistiques par feature et globales
    """
    perturbation = np.abs(X_adv - X_clean)

    # Statistiques globales
    global_stats = {
        'mean_perturbation'   : float(perturbation.mean()),
        'max_perturbation'    : float(perturbation.max()),
        'l2_norm_mean'        : float(np.linalg.norm(X_adv - X_clean, axis=1).mean()),
        'linf_norm_mean'      : float(np.abs(X_adv - X_clean).max(axis=1).mean()),
        'pct_features_changed': float((perturbation > 1e-8).mean() * 100),
    }

    # Perturbation par feature (tri décroissant)
    per_feature = {
        feature_names[i]: float(perturbation[:, i].mean())
        for i in range(len(feature_names))
    }
    per_feature_sorted = dict(
        sorted(per_feature.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        'global'     : global_stats,
        'per_feature': per_feature_sorted,
    }


def plot_perturbation_by_feature(
    X_clean: np.ndarray,
    X_adv_fgsm: np.ndarray,
    X_adv_pgd: np.ndarray,
    feature_names: List[str],
    model_name: str,
    output_dir: Path,
    top_n: int = 15
) -> None:
    """
    Graphique : perturbation moyenne par feature pour FGSM et PGD.

    Args:
        X_clean       : Données propres
        X_adv_fgsm    : Exemples adverses FGSM
        X_adv_pgd     : Exemples adverses PGD
        feature_names : Noms des features
        model_name    : Nom du modèle (pour le titre)
        output_dir    : Répertoire de sauvegarde
        top_n         : Nombre de features à afficher
    """
    n_features = len(feature_names)

    # Perturbation moyenne absolue par feature
    pert_fgsm = np.abs(X_adv_fgsm - X_clean).mean(axis=0)
    pert_pgd  = np.abs(X_adv_pgd  - X_clean).mean(axis=0)

    # Trier par perturbation PGD décroissante
    top_indices = np.argsort(pert_pgd)[::-1][:min(top_n, n_features)]
    top_names   = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(top_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, pert_fgsm[top_indices], width,
                   label='FGSM', color='#E67E22', alpha=0.85)
    bars2 = ax.bar(x + width / 2, pert_pgd[top_indices],  width,
                   label='PGD',  color='#E74C3C', alpha=0.85)

    ax.set_xlabel('Features', fontweight='bold', fontsize=12)
    ax.set_ylabel('Perturbation Moyenne Absolue', fontweight='bold', fontsize=12)
    ax.set_title(f'Perturbation par Feature — {model_name}',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(top_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f"perturbation_features_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  ✓ Graphique perturbation sauvegardé: {save_path}")


def save_adversarial_examples(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    feature_names: List[str],
    model_name: str,
    attack_name: str,
    output_dir: Path,
    n_examples: int = 10
) -> None:
    """
    Sauvegarde des exemples adverses dans un fichier CSV pour inspection.

    Args:
        X_clean      : Données propres
        X_adv        : Données adverses
        y_true       : Vraies étiquettes
        y_pred_clean : Prédictions sur données propres
        y_pred_adv   : Prédictions sur données adverses
        feature_names: Noms des features
        model_name   : Nom du modèle
        attack_name  : 'FGSM' ou 'PGD'
        output_dir   : Répertoire de sauvegarde
        n_examples   : Nombre d'exemples à sauvegarder
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas non disponible — sauvegarde CSV ignorée")
        return

    # Sélectionner les exemples où l'attaque a réussi (prédiction changée)
    fooled_mask = (y_pred_clean != y_pred_adv)
    fooled_indices = np.where(fooled_mask)[0]

    if len(fooled_indices) == 0:
        logger.info(f"  Aucun exemple trompé pour {model_name}/{attack_name}")
        return

    # Prendre au max n_examples exemples
    selected = fooled_indices[:min(n_examples, len(fooled_indices))]

    rows = []
    for idx in selected:
        row = {
            'vrai_label'   : int(y_true[idx]),
            'pred_propre'  : int(y_pred_clean[idx]),
            'pred_adverse' : int(y_pred_adv[idx]),
        }
        # Ajouter les valeurs propres et adverses pour chaque feature
        for j, feat in enumerate(feature_names):
            row[f'{feat}_propre']  = float(X_clean[idx, j])
            row[f'{feat}_adverse'] = float(X_adv[idx, j])
            row[f'{feat}_delta']   = float(X_adv[idx, j] - X_clean[idx, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    save_path = output_dir / f"examples_{model_name.lower().replace(' ', '_')}_{attack_name.lower()}.csv"
    df.to_csv(save_path, index=False, encoding='utf-8')
    logger.info(f"  ✓ {len(selected)} exemples adverses sauvegardés: {save_path}")
