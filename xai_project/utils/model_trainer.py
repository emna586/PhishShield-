"""
Entraînement des modèles
"""

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any
from pathlib import Path
from config import MODELS, MODELS_DIR
from utils import setup_logger

logger = setup_logger(__name__)


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                model_key: str) -> Any:
    """
    Entraîne un modèle
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        model_key: Clé du modèle ('xgboost', 'lightgbm', 'random_forest')
        
    Returns:
        Modèle entraîné
    """
    logger.info(f"Entraînement de {MODELS[model_key]['name']}...")
    
    params = MODELS[model_key]['params']
    
    if model_key == 'xgboost':
        model = XGBClassifier(**params)
    elif model_key == 'lightgbm':
        model = LGBMClassifier(**params)
    elif model_key == 'random_forest':
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Modèle inconnu: {model_key}")
    
    model.fit(X_train, y_train)
    
    logger.info(f"✓ {MODELS[model_key]['name']} entraîné")
    
    return model


def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Entraîne tous les modèles
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        
    Returns:
        Dict avec les modèles entraînés
    """
    logger.info("=" * 60)
    logger.info("ENTRAÎNEMENT DES MODÈLES")
    logger.info("=" * 60)
    
    models = {}
    for model_key in MODELS.keys():
        models[model_key] = train_model(X_train, y_train, model_key)
    
    logger.info("✓ Tous les modèles entraînés")
    logger.info("=" * 60)
    
    return models


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str) -> Dict[str, float]:
    """
    Évalue un modèle
    
    Args:
        model: Modèle à évaluer
        X_test: Features de test
        y_test: Target de test
        model_name: Nom du modèle pour logging
        
    Returns:
        Dict avec les métriques
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'cm': confusion_matrix(y_test, y_pred)
    }
    
    logger.info(f"{model_name}:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    return metrics


def evaluate_all_models(models: Dict[str, Any], X_test: np.ndarray, 
                       y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Évalue tous les modèles
    
    Args:
        models: Dict des modèles entraînés
        X_test: Features de test
        y_test: Target de test
        
    Returns:
        Dict avec les métriques pour chaque modèle
    """
    logger.info("=" * 60)
    logger.info("ÉVALUATION DES MODÈLES")
    logger.info("=" * 60)
    
    all_metrics = {}
    for model_key, model in models.items():
        model_name = MODELS[model_key]['name']
        all_metrics[model_key] = evaluate_model(model, X_test, y_test, model_name)
    
    logger.info("=" * 60)
    
    return all_metrics


def save_models(models: Dict[str, Any]) -> None:
    """
    Sauvegarde tous les modèles
    
    Args:
        models: Dict des modèles à sauvegarder
    """
    logger.info("Sauvegarde des modèles...")
    
    for model_key, model in models.items():
        save_path = MODELS_DIR / f"{model_key}_model.pkl"
        joblib.dump(model, save_path)
        logger.info(f"  ✓ {MODELS[model_key]['name']} sauvegardé: {save_path}")


def load_models() -> Dict[str, Any]:
    """
    Charge tous les modèles depuis le disque
    
    Returns:
        Dict des modèles chargés
    """
    logger.info("Chargement des modèles...")
    
    models = {}
    for model_key in MODELS.keys():
        load_path = MODELS_DIR / f"{model_key}_model.pkl"
        if load_path.exists():
            models[model_key] = joblib.load(load_path)
            logger.info(f"  ✓ {MODELS[model_key]['name']} chargé: {load_path}")
        else:
            logger.warning(f"  ✗ Fichier non trouvé: {load_path}")
    
    return models
