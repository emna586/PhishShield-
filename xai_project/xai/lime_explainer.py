"""
Module LIME - Local Interpretable Model-agnostic Explanations
"""

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from typing import Tuple, Optional
from utils import setup_logger

logger = setup_logger(__name__)


def calculate_lime(model, X_train: np.ndarray, X_test: np.ndarray,
                   feature_names: list, model_name: str,
                   num_samples: int = 10, num_features: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule LIME importance en agrégant les explications locales
    
    Args:
        model: Modèle entraîné
        X_train: Features d'entraînement (données de background)
        X_test: Features de test
        feature_names: Noms des features
        model_name: Nom du modèle pour logging
        num_samples: Nombre d'instances pour calculer LIME
        num_features: Nombre de features à considérer (None = tous)
        
    Returns:
        Tuple de (lime_importance, top10_indices)
            - lime_importance: Vector d'importance [n_features]
            - top10_indices: Indices des top 10 features
    """
    logger.info(f"Calcul LIME pour {model_name}...")
    
    n_features = len(feature_names)
    if num_features is None:
        num_features = n_features
    
    # Créer le LIME explainer
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=['Legitimate', 'Phishing'],
            mode='classification',
            random_state=42
        )
        logger.info(f"  ✓ LimeTabularExplainer créé")
    except Exception as e:
        logger.error(f"Erreur lors de la création du LimeExplainer: {e}")
        raise
    
    # Limiter le nombre de samples
    num_samples_actual = min(num_samples, len(X_test))
    indices = np.random.choice(len(X_test), num_samples_actual, replace=False)
    logger.info(f"  Calcul des explications pour {num_samples_actual} instances")
    
    # Initialiser le vecteur d'importance
    lime_importances = np.zeros(n_features)
    
    # Collecter LIME pour chaque instance
    for i, idx in enumerate(indices):
        try:
            # Expliquer l'instance
            exp = lime_explainer.explain_instance(
                X_test[idx],
                model.predict_proba,
                num_features=num_features
            )
            
            # Extraire les importances locales pour la classe 1 (phishing)
            local_exp = exp.local_exp.get(1, {})  # Classe 1 = phishing
            
            # Ajouter à l'importance globale (valeur absolue pour éviter annulation des signes)
            for feature_idx, importance in local_exp.items():
                lime_importances[feature_idx] += abs(importance)
            
            if (i + 1) % max(1, num_samples_actual // 3) == 0:
                logger.info(f"    Progression: {i+1}/{num_samples_actual}")
                
        except Exception as e:
            logger.warning(f"  Erreur pour instance {idx}: {e}")
            continue
    
    # Moyenner les importances
    lime_importances /= num_samples_actual
    
    # Top 10 features
    top10_indices = np.argsort(np.abs(lime_importances))[::-1][:10]
    
    logger.info(f"  Top 10 features LIME:")
    for rank, idx in enumerate(top10_indices, 1):
        logger.info(f"    {rank}. {feature_names[idx]}: {lime_importances[idx]:.4f}")
    
    return lime_importances, top10_indices


def get_lime_top_features(lime_importance: np.ndarray, feature_names: list,
                          top_n: int = 10) -> list:
    """
    Retourne les noms des top N features selon LIME
    
    Args:
        lime_importance: Vector d'importance LIME
        feature_names: Noms des features
        top_n: Nombre de top features à retourner
        
    Returns:
        Liste des noms des top features
    """
    top_indices = np.argsort(np.abs(lime_importance))[::-1][:top_n]
    return [feature_names[i] for i in top_indices]
