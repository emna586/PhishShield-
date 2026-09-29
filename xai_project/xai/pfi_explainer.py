"""
Module PFI - Permutation Feature Importance
"""

import numpy as np
from sklearn.inspection import permutation_importance
from typing import Tuple
from utils import setup_logger

logger = setup_logger(__name__)


def calculate_pfi(model, X_test: np.ndarray, y_test: np.ndarray,
                  feature_names: list, model_name: str,
                  n_repeats: int = 10, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule Permutation Feature Importance
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Target de test
        feature_names: Noms des features
        model_name: Nom du modèle pour logging
        n_repeats: Nombre de répétitions pour le calcul
        random_state: Seed pour reproducibilité
        
    Returns:
        Tuple de (pfi_importance, top10_indices)
            - pfi_importance: Vector d'importance [n_features]
            - top10_indices: Indices des top 10 features
    """
    logger.info(f"Calcul PFI pour {model_name}...")
    
    # Limiter la taille pour accélérer le calcul
    sample_size = 1000
    if len(X_test) > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[indices]
        y_sample = y_test[indices]
        logger.info(f"  Utilisation d'un echantillon de {sample_size} observations pour PFI (sur {len(X_test)})")
    else:
        X_sample = X_test
        y_sample = y_test
        
    try:
        # Calculer PFI
        pfi_result = permutation_importance(
            model,
            X_sample,
            y_sample,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='accuracy',
            n_jobs=-1  # Paralléliser
        )
        
        logger.info(f"  ✓ PFI calculé avec {n_repeats} répétitions")
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul de PFI: {e}")
        raise
    
    # Extraire les importances moyennes
    pfi_importance = pfi_result.importances_mean
    pfi_std = pfi_result.importances_std
    
    # Top 10 features
    top10_indices = np.argsort(pfi_importance)[::-1][:10]
    
    logger.info(f"  Top 10 features PFI:")
    for rank, idx in enumerate(top10_indices, 1):
        logger.info(f"    {rank}. {feature_names[idx]}: {pfi_importance[idx]:.4f} (±{pfi_std[idx]:.4f})")
    
    return pfi_importance, top10_indices


def get_pfi_top_features(pfi_importance: np.ndarray, feature_names: list,
                         top_n: int = 10) -> list:
    """
    Retourne les noms des top N features selon PFI
    
    Args:
        pfi_importance: Vector d'importance PFI
        feature_names: Noms des features
        top_n: Nombre de top features à retourner
        
    Returns:
        Liste des noms des top features
    """
    top_indices = np.argsort(pfi_importance)[::-1][:top_n]
    return [feature_names[i] for i in top_indices]
