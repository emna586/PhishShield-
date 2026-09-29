"""
Module SHAP - SHapley Additive exPlanations
"""

import shap

# Patch pour corriger le bug de shap avec les versions récentes de xgboost
try:
    import shap.explainers._tree as shap_tree
    original_decode = shap_tree.decode_ubjson_buffer
    
    def patched_decode(*args, **kwargs):
        res = original_decode(*args, **kwargs)
        try:
            learner = res.get("learner", {})
            param = learner.get("learner_model_param", {})
            if "base_score" in param:
                val = param["base_score"]
                if isinstance(val, str):
                    param["base_score"] = val.strip("[]")
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    param["base_score"] = str(val[0])
        except Exception:
            pass
        return res
        
    shap_tree.decode_ubjson_buffer = patched_decode
except Exception:
    pass

import numpy as np
from typing import Tuple, Optional
from utils import setup_logger

logger = setup_logger(__name__)


def calculate_shap(model, X_test: np.ndarray, feature_names: list,
                   model_name: str, sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule SHAP importance pour un modèle tree-based
    
    Args:
        model: Modèle entraîné (XGBoost, LightGBM, Random Forest)
        X_test: Features de test
        feature_names: Noms des features
        model_name: Nom du modèle pour logging
        sample_size: Nombre de samples à utiliser (None = tous)
        
    Returns:
        Tuple de (shap_importance, top10_indices, shap_values)
            - shap_importance: Vector d'importance [n_features]
            - top10_indices: Indices des top 10 features
            - shap_values: Matrice SHAP complète [n_samples, n_features]
    """
    logger.info(f"Calcul SHAP pour {model_name}...")
    
    # Utiliser sample si spécifié
    if sample_size is not None and len(X_test) > sample_size:
        indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[indices]
        logger.info(f"  Utilisation d'un sample de {sample_size} observations (sur {len(X_test)})")
    else:
        X_sample = X_test
        logger.info(f"  Utilisation de toutes les {len(X_test)} observations")
    
    # Créer le SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        logger.info(f"  ✓ TreeExplainer créé pour {model_name}")
    except Exception as e:
        logger.error(f"Erreur lors de la création du TreeExplainer: {e}")
        raise
    
    # Calculer SHAP values
    try:
        shap_values = explainer.shap_values(X_sample)
        logger.info(f"  ✓ SHAP values calculées")
    except Exception as e:
        logger.error(f"Erreur lors du calcul des SHAP values: {e}")
        raise
    
    # Gérer les formats de sortie différents
    if isinstance(shap_values, list):
        # Pour les modèles binaires, shap_values peut être une liste [neg_class, pos_class]
        shap_values_array = shap_values[1]  # Classe positive (phishing = 1)
        logger.info(f"  Format: liste (classe phishing = index 1)")
    else:
        # Pour les modèles multiclass ou certains tree models
        shap_values_array = shap_values
        logger.info(f"  Format: array")
    
    # Calculer l'importance moyenne absolue
    shap_importance = np.abs(shap_values_array).mean(axis=0)
    
    # Top 10 features
    top10_indices = np.argsort(shap_importance)[::-1][:10]
    
    logger.info(f"  Top 10 features SHAP:")
    for rank, idx in enumerate(top10_indices, 1):
        logger.info(f"    {rank}. {feature_names[idx]}: {shap_importance[idx]:.4f}")
    
    return shap_importance, top10_indices, shap_values_array


def get_shap_top_features(shap_importance: np.ndarray, feature_names: list,
                          top_n: int = 10) -> list:
    """
    Retourne les noms des top N features selon SHAP
    
    Args:
        shap_importance: Vector d'importance SHAP
        feature_names: Noms des features
        top_n: Nombre de top features à retourner
        
    Returns:
        Liste des noms des top features
    """
    top_indices = np.argsort(shap_importance)[::-1][:top_n]
    return [feature_names[i] for i in top_indices]
