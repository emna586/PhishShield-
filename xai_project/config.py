"""
Configuration globale du projet XAI Phishing Detection
"""

import os
from pathlib import Path

# Chemins principaux
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
XAI_DIR = PROJECT_ROOT / "xai"
UTILS_DIR = PROJECT_ROOT / "utils"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
LOGS_DIR = OUTPUTS_DIR / "logs"
EXPLANATIONS_DIR = PROJECT_ROOT / "explanations"

# Créer les répertoires s'ils n'existent pas
for directory in [MODELS_DIR, VISUALIZATIONS_DIR, LOGS_DIR, EXPLANATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration des modèles
MODELS = {
    'xgboost': {
        'name': 'XGBoost',
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42,
            'verbosity': 0,
            'eval_metric': 'logloss'
        }
    },
    'lightgbm': {
        'name': 'LightGBM',
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 7,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
    },
    'random_forest': {
        'name': 'Random Forest',
        'params': {
            'n_estimators': 200,
            'max_depth': 15,
            'random_state': 42,
            'n_jobs': -1
        }
    }
}

# Configuration XAI
XAI_CONFIG = {
    'shap': {
        'enabled': True,
        'sample_size': None,  # None = utiliser tous les échantillons
        'use_background': False
    },
    'lime': {
        'enabled': True,
        'num_samples': 10,  # Nombre d'instances pour calculer LIME
        'num_features_lime': None  # None = tous les features
    },
    'pfi': {
        'enabled': True,
        'n_repeats': 10,
        'random_state': 42
    }
}

# Configuration de validation
VALIDATION_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'cross_validation_folds': 5  # Pour futur utilisation
}

# Seuils de cohérence
COHERENCE_THRESHOLDS = {
    'shap_pfi_corr': 0.80,      # SHAP vs PFI corrélation
    'shap_lime_corr': 0.75,     # SHAP vs LIME corrélation
    'lime_pfi_corr': 0.70,      # LIME vs PFI corrélation
    'shap_pfi_overlap': 0.75,   # SHAP vs PFI Top-10 overlap
    'shap_lime_overlap': 0.75,  # SHAP vs LIME Top-10 overlap
    'lime_pfi_overlap': 0.70    # LIME vs PFI Top-10 overlap
}

# Configuration de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': LOGS_DIR / 'xai_project.log'
}

# Configuration des visualisations
VISUALIZATION_CONFIG = {
    'dpi': 300,
    'figsize_single': (12, 6),
    'figsize_multiple': (15, 10),
    'style': 'seaborn-v0_8-darkgrid',
    'colormap': 'RdYlGn'
}

# Noms de classes
CLASS_NAMES = ['Legitimate', 'Phishing']
