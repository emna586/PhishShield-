"""
FGSM Attack — Fast Gradient Sign Method
========================================
Attaque adversariale en une seule étape.
Adaptée aux modèles tree-based via approximation numérique du gradient.

Référence : Goodfellow et al. (2014) "Explaining and Harnessing Adversarial Examples"
"""

import numpy as np
from utils import setup_logger

logger = setup_logger(__name__)


def _compute_numerical_gradient(model, X_batch: np.ndarray, delta: float = 1e-5) -> np.ndarray:
    """
    Approxime le gradient de la probabilité phishing par différences finies.

    Pour les modèles tree-based (XGBoost, LightGBM, RF), il n'y a pas de gradient
    analytique. On utilise donc la méthode des différences centrées :
        ∂f/∂xj ≈ [f(x + δ·ej) - f(x - δ·ej)] / (2δ)

    Args:
        model   : Modèle entraîné avec méthode predict_proba
        X_batch : Array numpy (n_samples, n_features)
        delta   : Pas de perturbation infinitésimale

    Returns:
        grad : Gradient approximé (n_samples, n_features)
    """
    n_samples, n_features = X_batch.shape
    grad = np.zeros_like(X_batch, dtype=np.float64)

    for j in range(n_features):
        # Perturbation positive : x + δ·ej
        X_plus = X_batch.copy()
        X_plus[:, j] += delta

        # Perturbation négative : x - δ·ej
        X_minus = X_batch.copy()
        X_minus[:, j] -= delta

        # Probabilité de la classe phishing (classe 1) pour chaque sample
        prob_plus  = model.predict_proba(X_plus)[:, 1]   # shape (n_samples,)
        prob_minus = model.predict_proba(X_minus)[:, 1]  # shape (n_samples,)

        # Différences centrées — une colonne entière à la fois (vectorisé)
        grad[:, j] = (prob_plus - prob_minus) / (2 * delta)

    return grad


def fgsm_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    sample_size: int = 200,
    delta: float = 1e-5,
    random_state: int = 42
) -> dict:
    """
    Fast Gradient Sign Method (FGSM) — Attaque adversariale simple en une étape.

    Principe :
        X_adv = X + ε · sign(∇_x Loss)
        Pour la détection de phishing : on maximise la proba phishing
        pour les samples légitimes, et la proba légitime pour les phishing.

    Args:
        model        : Modèle entraîné (XGBoost, LightGBM, RF)
        X            : Données propres (n_samples, n_features)
        y            : Vraies étiquettes (n_samples,)
        epsilon      : Force de perturbation (défaut=0.1)
        sample_size  : Nombre de samples utilisés (pour la vitesse, défaut=200)
        delta        : Pas pour approximation numérique du gradient
        random_state : Seed pour reproductibilité

    Returns:
        dict avec :
            - 'X_adv'     : Exemples adverses (n_samples, n_features)
            - 'accuracy'  : Accuracy sous attaque FGSM
            - 'X_clean'   : Données propres utilisées
            - 'y_true'    : Vraies étiquettes utilisées
    """
    logger.info(f"[FGSM] Démarrage — epsilon={epsilon}, sample_size={sample_size}")

    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    # ── 1. Sous-échantillonnage pour la performance ──────────────────────────
    rng = np.random.RandomState(random_state)
    n_total = len(X)
    n_use   = min(sample_size, n_total)

    indices = rng.choice(n_total, n_use, replace=False)
    X_sub   = X[indices]
    y_sub   = y[indices]

    logger.info(f"[FGSM] Calcul du gradient sur {n_use}/{n_total} samples...")

    # ── 2. Approximation numérique du gradient ───────────────────────────────
    grad = _compute_numerical_gradient(model, X_sub, delta=delta)

    # ── 3. Signe du gradient — perturbation FGSM ─────────────────────────────
    sign_grad = np.sign(grad)
    X_adv = X_sub + epsilon * sign_grad

    # ── 4. Clipping dans les bornes des données originales ────────────────────
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    X_adv = np.clip(X_adv, x_min, x_max)

    # ── 5. Évaluation ─────────────────────────────────────────────────────────
    predictions = model.predict(X_adv)
    accuracy    = float(np.mean(predictions == y_sub))

    logger.info(f"[FGSM] Accuracy sous FGSM: {accuracy:.4f}")

    return {
        'X_adv'    : X_adv,
        'accuracy' : accuracy,
        'X_clean'  : X_sub,
        'y_true'   : y_sub,
    }
