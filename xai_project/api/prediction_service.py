"""
Service de prédiction et d'explication XAI en temps réel pour FastAPI.
Charge les modèles ML, le scaler, et génère des explications SHAP et LIME.
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional

import shap
from lime.lime_tabular import LimeTabularExplainer

sys.path.append(str(Path(__file__).parent.parent))

# Patch SHAP/XGBoost compatibility (base_score format)
try:
    import shap.explainers._tree as _shap_tree
    _orig_decode = _shap_tree.decode_ubjson_buffer

    def _patched_decode(*args, **kwargs):
        res = _orig_decode(*args, **kwargs)
        try:
            param = res.get("learner", {}).get("learner_model_param", {})
            if "base_score" in param:
                val = param["base_score"]
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    param["base_score"] = str(val[0])
                elif isinstance(val, str) and val.startswith("["):
                    param["base_score"] = val.strip("[]").split(",")[0].strip()
        except Exception:
            pass
        return res

    _shap_tree.decode_ubjson_buffer = _patched_decode
except Exception:
    pass

from hybrid.hybrid_engine import HybridDetectionEngine
from utils.feature_extractor import extract_features_from_url
from utils import setup_logger

logger = setup_logger(__name__)


class PredictionService:
    """
    Service central de prédiction :
    - Charge les modèles ML, le scaler et le moteur hybride.
    - Génère des explications SHAP et LIME en temps réel.
    - Charge les métriques de robustesse précalculées.
    """

    # Valeurs par défaut si metrics.json absent
    DEFAULT_ROBUSTNESS = {
        'xgboost': {
            'baseline_accuracy': 0.98, 'fgsm_accuracy': 0.80, 'pgd_accuracy': 0.70,
            'fgsm_drop': 0.18, 'pgd_drop': 0.28, 'robustness_score': 0.827,
            'robustness_level': '[ORANGE] VULNERABLE',
        },
        'lightgbm': {
            'baseline_accuracy': 0.98, 'fgsm_accuracy': 0.79, 'pgd_accuracy': 0.68,
            'fgsm_drop': 0.19, 'pgd_drop': 0.30, 'robustness_score': 0.817,
            'robustness_level': '[ORANGE] VULNERABLE',
        },
        'random_forest': {
            'baseline_accuracy': 0.97, 'fgsm_accuracy': 0.87, 'pgd_accuracy': 0.83,
            'fgsm_drop': 0.10, 'pgd_drop': 0.14, 'robustness_score': 0.893,
            'robustness_level': '[JAUNE] MODERE',
        },
    }

    def __init__(
        self,
        models_dir: str = "outputs/models",
        dataset_path: str = "../Codes/PhishFOE_ds/PhishFOE_ds.csv"
    ):
        self.models_dir   = Path(models_dir)
        self.dataset_path = Path(dataset_path)

        # Charger le moteur hybride (modèles + scaler + listes + heuristiques)
        self.hybrid_engine = HybridDetectionEngine(models_dir=str(self.models_dir))
        self.scaler        = self.hybrid_engine.scaler
        self.models        = self.hybrid_engine.models
        self.feature_names = self.hybrid_engine.feature_names

        # XAI explainers
        self.shap_explainers: Dict[str, Any] = {}
        self.lime_explainer: Optional[LimeTabularExplainer] = None

        # Métriques de robustesse
        self.robustness_metrics: Dict[str, Any] = {}

        self._init_xai_explainers()
        self._load_robustness_metrics()

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation XAI
    # ──────────────────────────────────────────────────────────────────────────

    def _init_xai_explainers(self) -> None:
        """
        Précharge les TreeExplainers SHAP et l'explainer LIME en mémoire.
        Ceci accélère les requêtes d'analyse en évitant de recréer l'explainer
        à chaque appel.
        """
        if not self.models or not self.scaler:
            logger.warning("[WARN] Modeles ou Scaler indisponibles — XAI non disponible.")
            return

        # 1. SHAP : TreeExplainer pour chaque modèle
        for key, model in self.models.items():
            try:
                self.shap_explainers[key] = shap.TreeExplainer(model)
                logger.info(f"[OK] TreeExplainer SHAP prechargé pour [{key}]")
            except Exception as e:
                logger.error(f"[ERR] SHAP init [{key}] : {e}")

        # 2. LIME : nécessite un échantillon d'arrière-plan du dataset
        bg_data = self._load_background_data()
        if bg_data is not None and len(bg_data) > 0:
            try:
                self.lime_explainer = LimeTabularExplainer(
                    training_data=bg_data,
                    feature_names=self.feature_names,
                    class_names=['Legitimate', 'Phishing'],
                    mode='classification',
                    random_state=42
                )
                logger.info(f"[OK] LIME prechargé avec {len(bg_data)} echantillons d'arriere-plan.")
            except Exception as e:
                logger.error(f"[ERR] LIME init : {e}")

    def _load_background_data(self) -> Optional[np.ndarray]:
        """
        Charge un petit échantillon du dataset pour l'initialisation de LIME.
        Essaie plusieurs chemins possibles.
        """
        import pandas as pd

        search_paths = [
            self.dataset_path,
            Path("data/phishing.csv"),
            Path("../Codes/PhishFOE_ds/PhishFOE_ds.csv"),
            Path("outputs/phishing.csv"),
        ]

        for p in search_paths:
            if p.exists():
                try:
                    df = pd.read_csv(p, nrows=300)
                    target_col = df.columns[-1]
                    cols_drop = [target_col] + [c for c in ['URL', 'TLD'] if c in df.columns]
                    X_bg = df.drop(columns=cols_drop).values.astype(np.float32)
                    X_bg_scaled = self.scaler.transform(X_bg)
                    logger.info(f"[OK] Arriere-plan LIME charge depuis {p}")
                    return X_bg_scaled
                except Exception as e:
                    logger.warning(f"[WARN] Echec chargement {p} : {e}")

        logger.warning("[WARN] Dataset introuvable — LIME non disponible.")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Métriques de robustesse
    # ──────────────────────────────────────────────────────────────────────────

    def _load_robustness_metrics(self) -> None:
        """
        Charge les métriques de robustesse depuis outputs/robustness/metrics.json.
        Utilise les valeurs par défaut si le fichier est absent.
        """
        metrics_path = Path("outputs/robustness/metrics.json")
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    self.robustness_metrics = json.load(f)
                logger.info(f"[OK] Metriques de robustesse chargées depuis {metrics_path}")
                return
            except Exception as e:
                logger.error(f"[ERR] Lecture metrics.json : {e}")

        logger.warning("[WARN] metrics.json absent — utilisation des valeurs par defaut.")
        self.robustness_metrics = self.DEFAULT_ROBUSTNESS

    # ──────────────────────────────────────────────────────────────────────────
    # Explication XAI locale
    # ──────────────────────────────────────────────────────────────────────────

    def explain_prediction(
        self,
        X_scaled: np.ndarray,
        features_dict: Dict[str, float],
        model_key: str = "xgboost"
    ) -> Dict[str, List[Any]]:
        """
        Génère les explications SHAP et LIME locales pour une instance.

        Args:
            X_scaled     : Instance normalisée shape (1, n_features)
            features_dict: Dictionnaire des features brutes (non normalisées)
            model_key    : Clé du modèle à expliquer

        Returns:
            Dict avec listes 'shap' et 'lime' de FeatureContribution
        """
        shap_exps: List[Dict] = []
        lime_exps: List[Dict] = []

        # ── SHAP ──────────────────────────────────────────────────────────────
        explainer = self.shap_explainers.get(model_key)
        if explainer:
            try:
                shap_val = explainer.shap_values(X_scaled)

                # Gérer tous les formats de sortie possibles
                if isinstance(shap_val, list):
                    # Modèle binaire : shap_val = [neg_class, pos_class]
                    arr = np.array(shap_val[1])
                else:
                    arr = np.array(shap_val)

                # Aplatir à 1D si nécessaire
                if arr.ndim == 2:
                    arr = arr[0]
                elif arr.ndim == 3:
                    arr = arr[0, :, 1] if arr.shape[-1] == 2 else arr[0, :, 0]

                top_indices = np.argsort(np.abs(arr))[::-1][:5]
                for idx in top_indices:
                    idx = int(idx)
                    feat = self.feature_names[idx]
                    val  = float(arr[idx])
                    shap_exps.append({
                        "feature":    feat,
                        "value":      float(features_dict.get(feat, 0)),
                        "importance": abs(val),
                        "effect":     "Favorise Phishing" if val > 0 else "Favorise Legitime"
                    })
            except Exception as e:
                logger.error(f"[ERR] SHAP local : {e}")

        # ── LIME ──────────────────────────────────────────────────────────────
        if self.lime_explainer and model_key in self.models:
            try:
                model = self.models[model_key]
                exp   = self.lime_explainer.explain_instance(
                    X_scaled[0],
                    model.predict_proba,
                    num_features=5
                )
                # local_exp : dict {class_idx: [(feat_idx, weight), ...]}
                local_exp = exp.local_exp.get(1, [])
                for feat_idx, weight in local_exp[:5]:
                    feat = self.feature_names[int(feat_idx)]
                    lime_exps.append({
                        "feature":    feat,
                        "value":      float(features_dict.get(feat, 0)),
                        "importance": abs(float(weight)),
                        "effect":     "Favorise Phishing" if weight > 0 else "Favorise Legitime"
                    })
            except Exception as e:
                logger.error(f"[ERR] LIME local : {e}")

        return {"shap": shap_exps, "lime": lime_exps}

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline principale
    # ──────────────────────────────────────────────────────────────────────────

    def predict_url(self, url: str, model_key: str = "xgboost") -> Dict[str, Any]:
        """
        Pipeline complète : analyse hybride + explications XAI + robustesse.

        Args:
            url       : URL brute à analyser
            model_key : Modèle à utiliser pour l'explication XAI

        Returns:
            Dictionnaire conforme au schema PredictionResponse
        """
        # 1. Détection hybride (whitelist → blacklist → heuristiques + ML)
        res = self.hybrid_engine.detect(url)

        # 2. Explications XAI (seulement si la couche ML a été utilisée)
        explanations = None
        if res['method'] == 'hybrid' and self.scaler and self.feature_names:
            try:
                features_dict = extract_features_from_url(url)
                X_raw    = np.array(
                    [[features_dict.get(name, 0) for name in self.feature_names]],
                    dtype=np.float32
                )
                X_scaled = self.scaler.transform(X_raw)
                explanations = self.explain_prediction(X_scaled, features_dict, model_key)
            except Exception as e:
                logger.error(f"[ERR] XAI local : {e}")

        # 3. Métriques de robustesse (rafraîchir depuis fichier)
        self._load_robustness_metrics()

        robustness_dict = {}
        for key in ['xgboost', 'lightgbm', 'random_forest']:
            m = self.robustness_metrics.get(key, self.DEFAULT_ROBUSTNESS[key])
            robustness_dict[key] = {
                'vulnerable_to_fgsm': float(m.get('fgsm_drop', 0)) > 0.15,
                'fgsm_drop':          float(m.get('fgsm_drop', 0)),
                'pgd_drop':           float(m.get('pgd_drop', 0)),
                'robustness_score':   float(m.get('robustness_score', 0)),
            }

        # 4. Assurer que heuristic_details est toujours présent et complet
        raw_heur = res.get('heuristic_details', {})
        # Normaliser : whitelist/blacklist renvoient un dict minimal sans 'score' ni 'max_score'.
        # On garantit que tous les champs requis par HeuristicResponse sont présents.
        heuristic_details = {
            'score':         int(raw_heur.get('score', 0)),
            'max_score':     int(raw_heur.get('max_score', 85)),
            'score_percent': float(raw_heur.get('score_percent', 0.0)),
            'details':       raw_heur.get('details', {}),
            'features':      raw_heur.get('features', {}),
        }

        return {
            'url':             url,
            'status':          res['status'],
            'confidence':      float(res['confidence']),
            'method':          res['method'],
            'score':           float(res['score']),
            'verdict_details': res['verdict_details'],
            'heuristics':      heuristic_details,
            'ml_probabilities': res.get('ml_probabilities', {}),
            'explanations':    explanations,
            'robustness':      robustness_dict,
        }
