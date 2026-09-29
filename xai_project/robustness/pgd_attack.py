"""
PGD Attack — Projected Gradient Descent
=========================================
Attaque adversariale itérative — la plus puissante contre les classifieurs.
Adaptée aux modèles tree-based via approximation numérique du gradient.

Référence : Madry et al. (2018) "Towards Deep Learning Models Resistant to Adversarial Attacks"
"""

import numpy as np
from utils import setup_logger
from .fgsm_attack import _compute_numerical_gradient

logger = setup_logger(__name__)


def pgd_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    steps: int = 40,
    alpha: float = 0.01,
    sample_size: int = 100,
    delta: float = 1e-5,
    random_state: int = 42
) -> dict:
    """
    Projected Gradient Descent (PGD) — Attaque adversariale itérative.

    Principe (Madry et al., 2018) :
        Initialiser X_adv = X (ou X + bruit aléatoire)
        Pour chaque itération t :
            X_adv = X_adv + α · sign(∇_x Loss(X_adv))
            X_adv = Proj_{B(X, ε)}(X_adv)     ← projeter dans la boule ε

    PGD est une version itérative de FGSM : chaque étape fait un petit pas
    dans la direction du gradient, puis on projette dans la boule de rayon ε.
    Avec assez d'étapes, PGD trouve le point le plus adversarial dans la boule.

    Args:
        model        : Modèle entraîné (XGBoost, LightGBM, RF)
        X            : Données propres (n_samples, n_features)
        y            : Vraies étiquettes (n_samples,)
        epsilon      : Rayon de la boule de perturbation (défaut=0.1)
        steps        : Nombre d'itérations (défaut=40)
        alpha        : Taille de pas par itération (défaut=0.01 = ε/10)
        sample_size  : Nombre de samples utilisés (défaut=100, PGD est lent)
        delta        : Pas pour approximation numérique du gradient
        random_state : Seed pour reproductibilité

    Returns:
        dict avec :
            - 'X_adv'            : Exemples adverses finaux (n_samples, n_features)
            - 'accuracy'         : Accuracy sous attaque PGD
            - 'X_clean'          : Données propres utilisées
            - 'y_true'           : Vraies étiquettes utilisées
            - 'accuracy_history' : Accuracy à chaque étape (pour analyse)
    """
    logger.info(f"[PGD] Démarrage — epsilon={epsilon}, steps={steps}, alpha={alpha}, sample_size={sample_size}")

    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    # ── 1. Sous-échantillonnage ──────────────────────────────────────────────
    rng = np.random.RandomState(random_state)
    n_total = len(X)
    n_use   = min(sample_size, n_total)

    indices = rng.choice(n_total, n_use, replace=False)
    X_clean = X[indices]
    y_sub   = y[indices]

    # Bornes des données originales (pour clipping final)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)

    # ── 2. Initialisation de X_adv ───────────────────────────────────────────
    # Initialiser avec un petit bruit aléatoire uniforme dans [-ε, +ε]
    # (meilleur point de départ que simplement X_clean)
    noise = rng.uniform(-epsilon, epsilon, size=X_clean.shape)
    X_adv = X_clean + noise
    X_adv = np.clip(X_adv, x_min, x_max)

    logger.info(f"[PGD] Calcul du gradient sur {n_use}/{n_total} samples × {steps} étapes...")

    # ── 3. Boucle PGD ────────────────────────────────────────────────────────
    accuracy_history = []

    for step in range(steps):

        # 3a. Approximer gradient ∇_x f(X_adv)
        grad = _compute_numerical_gradient(model, X_adv, delta=delta)

        # 3b. Mise à jour : X_adv ← X_adv + α · sign(grad)
        sign_grad = np.sign(grad)
        X_adv     = X_adv + alpha * sign_grad

        # 3c. Projection dans la boule ε centrée sur X_clean
        #     Proj_{B(X, ε)}(X_adv) = clip(X_adv, X_clean - ε, X_clean + ε)
        X_adv = np.clip(X_adv, X_clean - epsilon, X_clean + epsilon)

        # 3d. Clipping dans les bornes des données (contrainte globale)
        X_adv = np.clip(X_adv, x_min, x_max)

        # 3e. Loguer l'accuracy tous les 10 steps (sans surcharger les logs)
        if (step + 1) % 10 == 0 or step == 0:
            preds_step = model.predict(X_adv)
            acc_step   = float(np.mean(preds_step == y_sub))
            accuracy_history.append({'step': step + 1, 'accuracy': acc_step})
            logger.info(f"[PGD] Étape {step + 1:>3}/{steps} — Accuracy: {acc_step:.4f}")

    # ── 4. Évaluation finale ─────────────────────────────────────────────────
    predictions = model.predict(X_adv)
    accuracy    = float(np.mean(predictions == y_sub))

    logger.info(f"[PGD] Accuracy finale sous PGD: {accuracy:.4f}")

    return {
        'X_adv'            : X_adv,
        'accuracy'         : accuracy,
        'X_clean'          : X_clean,
        'y_true'           : y_sub,
        'accuracy_history' : accuracy_history,
    }
