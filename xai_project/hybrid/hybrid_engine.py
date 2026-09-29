"""
Moteur de Détection Hybride — Whitelist/Blacklist, Heuristiques et Machine Learning.
Coordonne 3 couches de décision pour une détection robuste et explicable.
"""

import sys
from pathlib import Path
import numpy as np
import joblib
from typing import Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent))

from hybrid.whitelist_blacklist import WhitelistBlacklist
from hybrid.heuristic_rules import HeuristicEngine
from utils.feature_extractor import extract_features_from_url
from utils import setup_logger

logger = setup_logger(__name__)


class HybridDetectionEngine:
    """
    Moteur principal coordonnant la détection de phishing à travers 3 couches :

    1. Listes (Whitelist / Blacklist) - Décisions immédiates à 100% de confiance.
    2. Règles Heuristiques - Analyse structurelle de l'URL (score de 0 à 100).
    3. Machine Learning - Ensemble de modèles pré-entraînés (XGBoost, LightGBM, RF).

    Résultat final = 70% ML + 30% Heuristiques (si ML disponible).
    """

    # Ordre exact des features attendues par les modèles ML
    FEATURE_NAMES = [
        'IsHTTPS', 'IsDomainIP', 'URLLength', 'NoOfSubDomain', 'NoOfDots',
        'NoOfObfuscatedChar', 'NoOfEqual', 'NoOfQmark', 'NoOfAmp', 'NoOfDigits',
        'LineLength', 'HasTitle', 'HasMeta', 'HasFavicon', 'HasExternalFormSubmit',
        'HasCopyright', 'HasSocialNetworking', 'HasPasswordField', 'HasSubmitButton',
        'HasKeywordBank', 'HasKeywordPay', 'HasKeywordCrypto', 'NoOfPopup',
        'NoOfiFrame', 'NoOfImage', 'NoOfJS', 'NoOfCSS', 'NoOfURLRedirect',
        'NoOfHyperlink'
    ]

    def __init__(
        self,
        models_dir: str = "outputs/models",
        whitelist_blacklist_file: str = "outputs/whitelist_blacklist.json"
    ):
        """
        Initialise le moteur hybride.

        Args:
            models_dir: Dossier contenant les modèles ML et le scaler.
            whitelist_blacklist_file: Fichier JSON de persistence des listes.
        """
        self.models_dir          = Path(models_dir)
        self.whitelist_blacklist  = WhitelistBlacklist(whitelist_blacklist_file)
        self.heuristic_engine     = HeuristicEngine()

        self.scaler        = None
        self.models        = {}
        self.feature_names = self.FEATURE_NAMES

        self.load_ml_components()

    # ------------------------------------------------------------------
    # Chargement des composants ML
    # ------------------------------------------------------------------

    def load_ml_components(self) -> None:
        """
        Tente de charger le scaler et les modèles ML depuis le dossier spécifié.
        En cas d'absence, le moteur se rabat sur les heuristiques seules.
        """
        # -- Scaler --
        scaler_path = self.models_dir / "scaler.joblib"
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"[OK] Scaler charge : {scaler_path}")
            except Exception as e:
                logger.error(f"[ERR] Scaler non charge : {e}")

        # -- Modèles --
        model_files = {
            'xgboost':       'xgboost_model.pkl',
            'lightgbm':      'lightgbm_model.pkl',
            'random_forest': 'random_forest_model.pkl',
        }

        for key, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    self.models[key] = joblib.load(model_path)
                    logger.info(f"[OK] Modele [{key}] charge : {model_path}")
                except Exception as e:
                    logger.error(f"[ERR] Modele {key} non charge : {e}")

        if self.models:
            logger.info(f"[OK] {len(self.models)} modele(s) ML disponible(s).")
        else:
            logger.warning("[WARN] Aucun modele ML disponible — mode heuristiques seules.")

    # ------------------------------------------------------------------
    # Pipeline de détection principale
    # ------------------------------------------------------------------

    def detect(self, url: str) -> Dict[str, Any]:
        """
        Exécute la pipeline de détection hybride complète sur une URL.

        Ordre de priorité :
            1. Whitelist  → légitime à 100%
            2. Blacklist  → phishing à 100%
            3. Heuristiques + ML → verdict combiné

        Args:
            url: URL brute à analyser.

        Returns:
            Dictionnaire contenant toutes les informations de décision.
        """
        url = url.strip()

        # ── Couche 1 : Whitelist ──────────────────────────────────────────────
        if self.whitelist_blacklist.check_whitelist(url):
            logger.info(f"[WHITELIST] {url}")
            return self._build_response(
                status='legitimate',
                confidence=1.0,
                method='whitelist',
                score=0.0,
                verdict_details="Domaine present dans la Whitelist (confiance absolue).",
                heuristic_result={'score_percent': 0.0, 'rules_triggered': [], 'rules_checked': []},
                ml_probs={}
            )

        # ── Couche 2 : Blacklist ──────────────────────────────────────────────
        if self.whitelist_blacklist.check_blacklist(url):
            logger.info(f"[BLACKLIST] {url}")
            return self._build_response(
                status='phishing',
                confidence=1.0,
                method='blacklist',
                score=100.0,
                verdict_details="Domaine present dans la Blacklist (confiance absolue).",
                heuristic_result={'score_percent': 100.0, 'rules_triggered': [], 'rules_checked': []},
                ml_probs={}
            )

        # ── Couche 3 : Heuristiques ───────────────────────────────────────────
        heur_result      = self.heuristic_engine.evaluate(url)
        heur_score_pct   = heur_result['score_percent']   # 0–100

        # ── Couche 4 : Machine Learning ───────────────────────────────────────
        ml_probs = {}
        has_ml   = False

        if self.scaler and self.models:
            try:
                features_dict = extract_features_from_url(url)
                X_raw    = np.array(
                    [[features_dict.get(name, 0) for name in self.feature_names]],
                    dtype=np.float32
                )
                X_scaled = self.scaler.transform(X_raw)

                for name, model in self.models.items():
                    prob = float(model.predict_proba(X_scaled)[0, 1])
                    ml_probs[name] = prob

                has_ml = True
                logger.info(f"[ML] Probas phishing : {ml_probs}")
            except Exception as e:
                logger.error(f"[ERR] Prediction ML echouee : {e}", exc_info=True)

        # ── Combinaison & Verdict Final ───────────────────────────────────────
        if has_ml:
            avg_ml_prob   = float(np.mean(list(ml_probs.values())))
            # 70% ML + 30% Heuristiques
            combined_score = (0.7 * avg_ml_prob + 0.3 * (heur_score_pct / 100.0)) * 100.0
            status         = 'phishing' if combined_score >= 50.0 else 'legitimate'
            # Confiance normalisée entre 50% et 100%
            raw_conf       = abs(combined_score - 50.0) / 50.0
            confidence     = 0.5 + raw_conf * 0.5
            verdict_details = (
                f"Analyse hybride (ML ensemble: {avg_ml_prob*100:.1f}%, "
                f"Heuristiques: {heur_score_pct:.1f}%)."
            )
            method = 'hybrid'
        else:
            # Fallback : heuristiques seules
            combined_score  = heur_score_pct
            status          = 'phishing' if combined_score >= 40.0 else 'legitimate'
            raw_conf        = abs(combined_score - 40.0) / 60.0
            confidence      = 0.5 + raw_conf * 0.5
            verdict_details = "Analyse par regles heuristiques uniquement (ML indisponible)."
            method          = 'heuristics'

        logger.info(f"[VERDICT] status={status}, confidence={confidence:.3f}, score={combined_score:.1f}")

        return self._build_response(
            status=status,
            confidence=float(confidence),
            method=method,
            score=float(combined_score),
            verdict_details=verdict_details,
            heuristic_result=heur_result,
            ml_probs=ml_probs
        )

    # ------------------------------------------------------------------
    # Méthode helper pour construire la réponse
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response(
        status: str,
        confidence: float,
        method: str,
        score: float,
        verdict_details: str,
        heuristic_result: Dict[str, Any],
        ml_probs: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Construit un dictionnaire de réponse uniforme pour toutes les couches.
        Garantit que toutes les clés attendues par l'API sont toujours présentes.
        """
        return {
            'status':           status,
            'confidence':       confidence,
            'method':           method,
            'score':            score,
            'verdict_details':  verdict_details,
            'heuristic_details': heuristic_result,
            'ml_probabilities': ml_probs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide (exécution directe)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test local du moteur hybride...")
    engine = HybridDetectionEngine()

    test_urls = [
        "https://www.google.com",
        "https://accounts.google.com/login",
        "http://phishing-bank.com/secure/login",
        "http://123.45.67.89/update-details?id=992&bank=paypal",
        "https://secure-bank-login.com/signin",
        "https://paypal.com/secure/login",
    ]

    for url in test_urls:
        print("\n" + "=" * 60)
        print(f"Analyse : {url}")
        res = engine.detect(url)
        print(f"Verdict  : {res['status'].upper()} ({res['confidence']*100:.1f}%)")
        print(f"Methode  : {res['method']}")
        print(f"Score    : {res['score']:.1f}/100")
        print(f"Details  : {res['verdict_details']}")
