"""
Visualisations pour XAI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
from config import VISUALIZATION_CONFIG, VISUALIZATIONS_DIR, MODELS
from utils import setup_logger

logger = setup_logger(__name__)

# Set style (seaborn-v0_8-darkgrid est un style matplotlib, pas seaborn)
try:
    plt.style.use(VISUALIZATION_CONFIG['style'])
except OSError:
    sns.set_theme(style='darkgrid')


def plot_feature_importance_comparison(importances_dict: Dict[str, Dict[str, np.ndarray]],
                                       feature_names: List[str],
                                       top_n: int = 15) -> None:
    """
    Compare les 3 méthodes XAI pour un modèle donné
    
    Args:
        importances_dict: Dict avec SHAP, LIME, PFI importances
        feature_names: Noms des features
        top_n: Nombre de top features à afficher
    """
    logger.info("Génération des graphiques de comparaison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Comparaison des 3 méthodes XAI - Top 15 Features', fontsize=16, fontweight='bold')
    
    methods = {
        'SHAP': importances_dict.get('shap', np.zeros(len(feature_names))),
        'LIME': importances_dict.get('lime', np.zeros(len(feature_names))),
        'PFI': importances_dict.get('pfi', np.zeros(len(feature_names)))
    }
    
    for ax, (method_name, importances) in zip(axes, methods.items()):
        # Top N indices
        top_indices = np.argsort(np.abs(importances))[::-1][:top_n]
        top_names = [feature_names[i] for i in top_indices]
        top_values = importances[top_indices]
        
        # Plot
        colors = ['green' if v > 0 else 'red' for v in top_values]
        ax.barh(range(len(top_names)), np.abs(top_values), color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel('Importance (absolute value)', fontweight='bold')
        ax.set_title(f'{method_name}', fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_coherence_heatmap(coherence_data: Dict[str, float], model_name: str) -> None:
    """
    Visualise la matrice de cohérence
    
    Args:
        coherence_data: Dict avec données de cohérence
        model_name: Nom du modèle
    """
    logger.info(f"Génération de la heatmap de cohérence pour {model_name}...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Créer une matrice 3x3 avec corrélations
    methods = ['SHAP', 'LIME', 'PFI']
    corr_matrix = np.ones((3, 3))
    
    corr_matrix[0, 1] = coherence_data.get('SHAP_LIME_corr', 0)
    corr_matrix[1, 0] = coherence_data.get('SHAP_LIME_corr', 0)
    corr_matrix[0, 2] = coherence_data.get('SHAP_PFI_corr', 0)
    corr_matrix[2, 0] = coherence_data.get('SHAP_PFI_corr', 0)
    corr_matrix[1, 2] = coherence_data.get('LIME_PFI_corr', 0)
    corr_matrix[2, 1] = coherence_data.get('LIME_PFI_corr', 0)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=methods, yticklabels=methods, ax=ax, cbar_kws={'label': 'Correlation'})
    
    ax.set_title(f'Matrice de Cohérence - {model_name}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    return fig, ax


def plot_overlap_analysis(overlaps_dict: Dict[str, float], model_name: str) -> None:
    """
    Visualise l'analyse d'overlap des Top-10 features
    
    Args:
        overlaps_dict: Dict avec données d'overlap
        model_name: Nom du modèle
    """
    logger.info(f"Génération du graphique d'overlap pour {model_name}...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    comparisons = ['SHAP vs LIME', 'SHAP vs PFI', 'LIME vs PFI']
    overlaps = [
        overlaps_dict.get('SHAP_LIME_overlap', 0),
        overlaps_dict.get('SHAP_PFI_overlap', 0),
        overlaps_dict.get('LIME_PFI_overlap', 0)
    ]
    
    thresholds = [75, 75, 70]  # Seuils attendus
    
    x = np.arange(len(comparisons))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, overlaps, width, label='Overlap %', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, thresholds, width, label='Threshold', color='orange', alpha=0.8)
    
    ax.axhline(y=75, color='r', linestyle='--', alpha=0.3, label='Target')
    ax.set_xlabel('Comparaisons', fontweight='bold')
    ax.set_ylabel('Overlap (%)', fontweight='bold')
    ax.set_title(f'Analyse d\'Overlap Top-10 Features - {model_name}', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(comparisons)
    ax.legend()
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    return fig, ax


def save_figure(fig, filename: str, format: str = 'png') -> None:
    """
    Sauvegarde une figure
    
    Args:
        fig: Figure matplotlib à sauvegarder
        filename: Nom du fichier (sans extension)
        format: Format de fichier ('png', 'pdf', 'jpg')
    """
    save_path = VISUALIZATIONS_DIR / f"{filename}.{format}"
    fig.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    logger.info(f"  ✓ Figure sauvegardée: {save_path}")
    plt.close(fig)
