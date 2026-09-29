"""
Chargement et préparation des données
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from utils import setup_logger

logger = setup_logger(__name__)


def load_phishing_dataset(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, List[str], np.ndarray]:
    """
    Charge le dataset de phishing du fichier CSV
    
    Args:
        csv_path: Chemin du fichier CSV
        
    Returns:
        Tuple de (X, y, feature_names, scaler)
    """
    logger.info(f"Chargement du dataset depuis {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Vérifier et traiter les valeurs manquantes
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"⚠ {nan_count} valeurs manquantes détectées — imputation par médiane")
        df = df.fillna(df.median(numeric_only=True))
        logger.info("✓ Imputation des NaN terminée")
    else:
        logger.info("✓ Aucune valeur manquante détectée")
    
    # Dernière colonne = target
    target_col = df.columns[-1]
    
    # Séparer features et target en excluant les colonnes textuelles (URL, TLD)
    cols_to_drop = [target_col]
    for col in ['URL', 'TLD']:
        if col in df.columns:
            cols_to_drop.append(col)
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col].values
    feature_names = X.columns.tolist()
    
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Classes: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    # Convertir en numpy
    X = X.values
    
    # Vérifier et corriger les types
    if X.dtype == object:
        X = X.astype(np.float32)
    
    logger.info(f"Shape final: X={X.shape}, y={y.shape}")
    
    return df, X, y, feature_names


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = 42, stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divise les données en train/test
    
    Args:
        X: Features
        y: Target
        test_size: Proportion du test set
        random_state: Seed pour reproducibilité
        stratify: Si True, utilise stratified split
        
    Returns:
        Tuple de (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Division des données: test_size={test_size}, stratify={stratify}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"Train classes: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    logger.info(f"Test classes: 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalise les features avec StandardScaler
    
    Args:
        X_train: Features d'entraînement
        X_test: Features de test
        
    Returns:
        Tuple de (X_train_scaled, X_test_scaled, scaler)
    """
    logger.info("Normalisation des features avec StandardScaler")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Scaled X_train: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    logger.info(f"Scaled X_test: mean={X_test_scaled.mean():.4f}, std={X_test_scaled.std():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(csv_path: str, test_size: float = 0.2, 
                random_state: int = 42) -> dict:
    """
    Pipeline complet de préparation des données
    
    Args:
        csv_path: Chemin du fichier CSV
        test_size: Proportion du test set
        random_state: Seed pour reproducibilité
        
    Returns:
        Dict avec X_train, X_test, y_train, y_test, feature_names, scaler
    """
    logger.info("=" * 60)
    logger.info("PRÉPARATION DES DONNÉES")
    logger.info("=" * 60)
    
    # Charger
    df, X, y, feature_names = load_phishing_dataset(csv_path)
    
    # Diviser
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Normaliser
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    logger.info("✓ Préparation des données complétée")
    logger.info("=" * 60)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'df': df
    }
