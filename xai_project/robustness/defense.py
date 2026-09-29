"""
Defense Module — Amélioration de la Robustesse
===============================================
Implémente deux défenses contre les attaques adversariales :

  1. Feature Squeezing   : Arrondit / tronque les features pour annuler
                           les petites perturbations numériques.

  2. Adversarial Training: Augmente le dataset d'entraînement avec des
                           exemples FGSM générés à la volée, puis
                           réentraîne chaque modèle sur ce dataset enrichi.

Usage typique (standalone) :
    python -m robustness.defense
"""

import sys
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent.parent))

from config import MODELS, MODELS_DIR, OUTPUTS_DIR
from utils import setup_logger
from robustness.fgsm_attack import fgsm_attack
from robustness.pgd_attack  import pgd_attack

logger = setup_logger(__name__)

ROBUSTNESS_DIR = OUTPUTS_DIR / "robustness"
DEFENSE_DIR    = ROBUSTNESS_DIR / "defense"
DEFENSE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DÉFENSE 1 — FEATURE SQUEEZING
# ============================================================================

class FeatureSqueezer:
    """
    Applique le Feature Squeezing avant chaque prédiction.

    Principe :
        Les features binaires (0/1) et entières sont arrondies à leur valeur
        naturelle. Une perturbation adversariale de ±0.1 sur une feature
        entière devient nulle après arrondi → l'attaque est annulée.

    Stratégie par type de feature :
        - Binaire  (min=0, max=1, valeurs uniques=2) → round() → {0, 1}
        - Entière  (toutes valeurs entières)          → round() → entier
        - Continue                                    → garder tel quel
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.feature_types: List[str] = []   # 'binary', 'integer', 'continuous'
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'FeatureSqueezer':
        """
        Détermine automatiquement le type de chaque feature à partir des données.
        """
        self.feature_types = []
        for j in range(X.shape[1]):
            col = X[:, j]
            unique_vals = np.unique(col)
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                self.feature_types.append('binary')
            elif np.allclose(col, col.astype(int)):
                self.feature_types.append('integer')
            else:
                self.feature_types.append('continuous')
        self._fitted = True
        counts = {t: self.feature_types.count(t) for t in ['binary', 'integer', 'continuous']}
        logger.info(f"[FeatureSqueezer] Types detectes: {counts}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applique le squeezing sur X.
        """
        if not self._fitted:
            raise RuntimeError("Appeler .fit() avant .transform()")
        X_sq = X.copy().astype(np.float64)
        for j, ftype in enumerate(self.feature_types):
            if ftype in ('binary', 'integer'):
                X_sq[:, j] = np.round(X_sq[:, j])
        return X_sq

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def summary(self) -> Dict[str, Any]:
        if not self._fitted:
            return {}
        counts = {t: self.feature_types.count(t) for t in ['binary', 'integer', 'continuous']}
        squeezed_features = [
            self.feature_names[j]
            for j, ft in enumerate(self.feature_types)
            if ft in ('binary', 'integer')
        ]
        return {
            'total_features': len(self.feature_names),
            'squeezed_features_count': len(squeezed_features),
            'continuous_features_count': counts.get('continuous', 0),
            'squeezed_feature_names': squeezed_features,
            'type_counts': counts,
        }


# ============================================================================
# DÉFENSE 2 — ADVERSARIAL TRAINING
# ============================================================================

def generate_adversarial_augmentation(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 0.1,
    augmentation_ratio: float = 0.30,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère des exemples adversariaux FGSM et les ajoute au dataset d'entraînement.

    Args:
        model             : Modèle entraîné (sur lequel générer les exemples)
        X_train           : Features d'entraînement originales
        y_train           : Labels d'entraînement originaux
        epsilon           : Force de perturbation FGSM
        augmentation_ratio: Proportion du dataset à augmenter (défaut 30%)
        random_state      : Seed pour reproductibilité

    Returns:
        X_aug, y_aug : Dataset augmenté (original + exemples adversariaux)
    """
    n_augment = int(len(X_train) * augmentation_ratio)
    n_augment = min(n_augment, 500)   # Cap pour la vitesse

    logger.info(f"[AdvTrain] Generation de {n_augment} exemples adversariaux FGSM (epsilon={epsilon})")

    result = fgsm_attack(
        model, X_train, y_train,
        epsilon=epsilon,
        sample_size=n_augment,
        random_state=random_state,
    )

    X_adv = result['X_adv']
    y_adv = result['y_true']

    X_aug = np.vstack([X_train, X_adv])
    y_aug = np.concatenate([y_train, y_adv])

    logger.info(f"[AdvTrain] Dataset augmente: {len(X_train)} -> {len(X_aug)} samples")
    return X_aug, y_aug


def adversarial_training(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 0.1,
    augmentation_ratio: float = 0.30,
) -> Dict[str, Any]:
    """
    Réentraîne chaque modèle sur un dataset augmenté avec des exemples adversariaux.

    Pipeline pour chaque modèle :
        1. Générer des exemples FGSM à partir du modèle actuel
        2. Combiner : X_aug = X_train + X_adv
        3. Réentraîner le modèle sur X_aug
        4. Sauvegarder sous outputs/models/<name>_robust_model.pkl

    Args:
        models            : Dict des modèles entraînés classiquement
        X_train           : Features d'entraînement normalisées
        y_train           : Labels d'entraînement
        epsilon           : Force de perturbation pour la génération FGSM
        augmentation_ratio: Fraction du train set à augmenter

    Returns:
        robust_models : Dict des modèles réentraînés
    """
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier

    robust_models = {}

    for model_key, model_obj in models.items():
        model_name = MODELS[model_key]['name']
        logger.info(f"\n{'='*60}")
        logger.info(f"  Adversarial Training : {model_name}")
        logger.info(f"{'='*60}")

        # -- Étape 1 : Augmenter les données --
        X_aug, y_aug = generate_adversarial_augmentation(
            model_obj, X_train, y_train,
            epsilon=epsilon,
            augmentation_ratio=augmentation_ratio,
        )

        # -- Étape 2 : Recréer et réentraîner un nouveau modèle --
        params = MODELS[model_key]['params']

        if model_key == 'xgboost':
            new_model = XGBClassifier(**params)
        elif model_key == 'lightgbm':
            new_model = LGBMClassifier(**params)
        elif model_key == 'random_forest':
            new_model = RandomForestClassifier(**params)
        else:
            logger.error(f"Modele inconnu: {model_key}")
            continue

        logger.info(f"[AdvTrain] Reentrainement de {model_name} sur {len(X_aug)} samples...")
        new_model.fit(X_aug, y_aug)
        logger.info(f"[AdvTrain] {model_name} reentrainé avec succes.")

        # -- Étape 3 : Sauvegarder --
        save_path = MODELS_DIR / f"{model_key}_robust_model.pkl"
        joblib.dump(new_model, save_path)
        logger.info(f"[AdvTrain] Modele sauvegarde: {save_path}")

        robust_models[model_key] = new_model

    return robust_models


# ============================================================================
# ÉVALUATION COMPARATIVE (Avant / Après défenses)
# ============================================================================

def evaluate_defense(
    original_models: Dict[str, Any],
    robust_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    squeezer: FeatureSqueezer,
    fgsm_epsilon: float = 0.1,
    pgd_epsilon: float  = 0.1,
    fgsm_sample: int    = 150,
    pgd_sample:  int    = 80,
) -> Dict[str, Any]:
    """
    Compare la robustesse AVANT et APRÈS les deux défenses.

    Pour chaque modèle, calcule :
        - Accuracy baseline (propre)
        - Accuracy sous FGSM / PGD — SANS défense
        - Accuracy sous FGSM / PGD — AVEC Feature Squeezing
        - Accuracy sous FGSM / PGD — AVEC Adversarial Training
        - Accuracy sous FGSM / PGD — AVEC les deux défenses combinées

    Returns:
        comparison : Dict complet des résultats
    """
    logger.info("\n" + "="*70)
    logger.info("  EVALUATION COMPARATIVE — AVANT vs APRES DEFENSES")
    logger.info("="*70)

    X_test_sq = squeezer.transform(X_test)
    comparison = {}

    for model_key in original_models:
        model_name  = MODELS[model_key]['name']
        orig_model  = original_models[model_key]
        rob_model   = robust_models.get(model_key)

        logger.info(f"\n{'─'*60}")
        logger.info(f"  Modele: {model_name}")
        logger.info(f"{'─'*60}")

        # -- Baseline --
        base_acc = float(orig_model.score(X_test, y_test))
        logger.info(f"  Baseline (propre)         : {base_acc:.4f}")

        # Helper local pour mesurer l'accuracy sous attaque
        def _eval(model, X_clean, label):
            r_fgsm = fgsm_attack(model, X_clean, y_test,
                                 epsilon=fgsm_epsilon, sample_size=fgsm_sample)
            r_pgd  = pgd_attack(model, X_clean, y_test,
                                epsilon=pgd_epsilon, steps=20, sample_size=pgd_sample)
            logger.info(f"  {label:42s} FGSM={r_fgsm['accuracy']:.4f}  PGD={r_pgd['accuracy']:.4f}")
            return r_fgsm['accuracy'], r_pgd['accuracy']

        # Scénario A : Sans défense
        fgsm_none, pgd_none = _eval(orig_model, X_test, "Sans defense")

        # Scénario B : Feature Squeezing uniquement
        fgsm_sq, pgd_sq = _eval(orig_model, X_test_sq, "Avec Feature Squeezing")

        # Scénario C : Adversarial Training uniquement
        fgsm_at, pgd_at = (None, None)
        if rob_model:
            fgsm_at, pgd_at = _eval(rob_model, X_test, "Avec Adversarial Training")

        # Scénario D : Les deux combinées
        fgsm_both, pgd_both = (None, None)
        if rob_model:
            fgsm_both, pgd_both = _eval(rob_model, X_test_sq, "Avec AT + Feature Squeezing")

        comparison[model_key] = {
            'model_name'  : model_name,
            'baseline'    : base_acc,
            'no_defense'  : {'fgsm': fgsm_none, 'pgd': pgd_none},
            'squeezing'   : {'fgsm': fgsm_sq,   'pgd': pgd_sq},
            'adv_training': {'fgsm': fgsm_at,   'pgd': pgd_at},
            'combined'    : {'fgsm': fgsm_both,  'pgd': pgd_both},
        }

    return comparison


# ============================================================================
# VISUALISATION COMPARATIVE
# ============================================================================

def plot_defense_comparison(comparison: Dict[str, Any]) -> None:
    """
    Génère un graphique comparatif côte-à-côte pour FGSM et PGD,
    montrant l'accuracy sous attaque pour chaque scénario de défense.
    """
    model_keys  = list(comparison.keys())
    model_names = [comparison[k]['model_name'] for k in model_keys]

    scenarios = {
        'Sans defense':         ('no_defense',   '#E74C3C'),
        'Feature Squeezing':    ('squeezing',     '#E67E22'),
        'Adversarial Training': ('adv_training',  '#3498DB'),
        'AT + Squeezing':       ('combined',      '#2ECC71'),
    }

    for attack_type in ('fgsm', 'pgd'):
        fig, ax = plt.subplots(figsize=(13, 7))
        x     = np.arange(len(model_names))
        n_sc  = len(scenarios)
        width = 0.18
        offset = -(n_sc - 1) / 2 * width

        for i, (label, (key, color)) in enumerate(scenarios.items()):
            values = []
            for mk in model_keys:
                v = comparison[mk].get(key, {})
                val = v.get(attack_type) if v else None
                values.append(val if val is not None else 0.0)

            bars = ax.bar(
                x + offset + i * width, values, width,
                label=label, color=color, alpha=0.88
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

        # Ligne baseline
        baseline_vals = [comparison[mk]['baseline'] for mk in model_keys]
        ax.hlines(baseline_vals, x - 0.4, x + 0.4,
                  colors='black', linestyles='--', linewidth=1.5,
                  label='Baseline (propre)')

        ax.set_xlabel('Modele', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy sous attaque', fontweight='bold', fontsize=12)
        ax.set_title(
            f'Comparaison des Defenses — Attaque {attack_type.upper()} (epsilon=0.1)',
            fontweight='bold', fontsize=14
        )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_ylim([0, 1.12])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        path = DEFENSE_DIR / f"defense_comparison_{attack_type}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"[OK] Graphique sauvegarde: {path}")

    # -- Graphique d'amélioration (gain en %) --
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax_idx, attack_type in enumerate(('fgsm', 'pgd')):
        ax = axes[ax_idx]
        x = np.arange(len(model_names))
        improvements = {
            'Feature Squeezing':    [],
            'Adversarial Training': [],
            'AT + Squeezing':       [],
        }
        for mk in model_keys:
            base_attack = comparison[mk]['no_defense'].get(attack_type, 0) or 0
            for label, key in [
                ('Feature Squeezing',    'squeezing'),
                ('Adversarial Training', 'adv_training'),
                ('AT + Squeezing',       'combined'),
            ]:
                defended = comparison[mk].get(key, {}) or {}
                val = defended.get(attack_type) or base_attack
                gain = (val - base_attack) * 100
                improvements[label].append(gain)

        colors_imp = {'Feature Squeezing': '#E67E22',
                      'Adversarial Training': '#3498DB',
                      'AT + Squeezing': '#2ECC71'}
        width = 0.25
        for i, (label, gains) in enumerate(improvements.items()):
            bars = ax.bar(x + (i - 1) * width, gains, width,
                          label=label, color=colors_imp[label], alpha=0.88)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + (0.3 if h >= 0 else -1.5),
                        f'+{h:.1f}%' if h >= 0 else f'{h:.1f}%',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(f'Gain de robustesse — {attack_type.upper()}',
                     fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel('Gain (points de %)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Amelioration de la Robustesse par Defenses',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = DEFENSE_DIR / "defense_gain.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"[OK] Graphique gain sauvegarde: {path}")


# ============================================================================
# RAPPORT TEXTE
# ============================================================================

def print_defense_report(comparison: Dict[str, Any],
                         squeezer_summary: Dict[str, Any]) -> str:
    """
    Génère et affiche un rapport comparatif lisible.
    Retourne le rapport en tant que string (pour sauvegarde).
    """
    lines = []

    def log(s=''):
        lines.append(s)
        print(s)

    log("=" * 70)
    log("  RAPPORT DE ROBUSTESSE — AVANT vs APRES DEFENSES")
    log("=" * 70)

    log(f"\n[Feature Squeezer]")
    log(f"  Features totales   : {squeezer_summary.get('total_features', '?')}")
    log(f"  Features squeezees : {squeezer_summary.get('squeezed_features_count', '?')}")
    log(f"  Features continues : {squeezer_summary.get('continuous_features_count', '?')}")

    for model_key, res in comparison.items():
        mname = res['model_name']
        log(f"\n{'─' * 70}")
        log(f"  {mname}")
        log(f"{'─' * 70}")
        log(f"  Baseline (propre)         : {res['baseline']:.4f}  ({res['baseline']*100:.2f}%)")

        def row(label, key):
            d = res.get(key, {}) or {}
            fgsm = d.get('fgsm')
            pgd  = d.get('pgd')
            fgsm_s = f"{fgsm:.4f}" if fgsm is not None else "N/A"
            pgd_s  = f"{pgd:.4f}"  if pgd  is not None else "N/A"
            log(f"  {label:40s} FGSM={fgsm_s}  PGD={pgd_s}")

        row("Sans defense",              'no_defense')
        row("Feature Squeezing",         'squeezing')
        row("Adversarial Training",      'adv_training')
        row("AT + Feature Squeezing",    'combined')

        # Meilleur résultat
        best_pgd_key  = None
        best_pgd_val  = -1.0
        for key in ('no_defense', 'squeezing', 'adv_training', 'combined'):
            v = (res.get(key) or {}).get('pgd')
            if v is not None and v > best_pgd_val:
                best_pgd_val = v
                best_pgd_key = key
        if best_pgd_key and best_pgd_key != 'no_defense':
            gain = best_pgd_val - (res.get('no_defense', {}).get('pgd') or 0)
            log(f"\n  >> Meilleure defense (PGD): [{best_pgd_key}] -> gain +{gain*100:.1f}%")

    log("\n" + "=" * 70)
    log("  Visualisations sauvegardees dans : outputs/robustness/defense/")
    log("=" * 70)

    report_str = "\n".join(lines)

    # Sauvegarde JSON
    report_path = DEFENSE_DIR / "defense_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        # Nettoyer les None pour JSON
        clean = {}
        for mk, res in comparison.items():
            clean[mk] = {}
            for k, v in res.items():
                if isinstance(v, dict):
                    clean[mk][k] = {kk: (float(vv) if vv is not None else None)
                                    for kk, vv in v.items()}
                else:
                    clean[mk][k] = float(v) if isinstance(v, float) else v
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info(f"[OK] Rapport JSON sauvegarde: {report_path}")

    return report_str


# ============================================================================
# PIPELINE PRINCIPALE (exécution autonome)
# ============================================================================

def run_defense_pipeline(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    epsilon: float = 0.1,
    augmentation_ratio: float = 0.30,
) -> Dict[str, Any]:
    """
    Pipeline complète :
        1. Entraîner le Feature Squeezer
        2. Adversarial Training pour chaque modèle
        3. Évaluation comparative Avant / Après
        4. Visualisations
        5. Rapport

    Returns:
        Dict avec 'robust_models', 'squeezer', 'comparison'
    """
    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE DEFENSE ADVERSARIALE")
    logger.info("  Techniques: Feature Squeezing + Adversarial Training")
    logger.info("=" * 70)

    # -- 1. Feature Squeezer --
    logger.info("\n[ETAPE 1] Feature Squeezing — Analyse des features")
    squeezer = FeatureSqueezer(feature_names)
    squeezer.fit(X_train)
    summary = squeezer.summary()
    logger.info(f"  {summary['squeezed_features_count']}/{summary['total_features']} features seront squeezees")

    # -- 2. Adversarial Training --
    logger.info("\n[ETAPE 2] Adversarial Training — Reentrainement sur dataset augmente")
    robust_models = adversarial_training(
        models, X_train, y_train,
        epsilon=epsilon,
        augmentation_ratio=augmentation_ratio,
    )

    # -- 3. Evaluation comparative --
    logger.info("\n[ETAPE 3] Evaluation comparative — Avant vs Apres")
    comparison = evaluate_defense(
        original_models=models,
        robust_models=robust_models,
        X_test=X_test,
        y_test=y_test,
        squeezer=squeezer,
    )

    # -- 4. Visualisations --
    logger.info("\n[ETAPE 4] Generation des visualisations")
    plot_defense_comparison(comparison)

    # -- 5. Rapport --
    logger.info("\n[ETAPE 5] Rapport final")
    print_defense_report(comparison, summary)

    return {
        'robust_models': robust_models,
        'squeezer':      squeezer,
        'comparison':    comparison,
    }


# ============================================================================
# POINT D'ENTRÉE STANDALONE
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    from utils.data_loader import prepare_data
    from utils.model_trainer import load_models

    logger.info("Lancement standalone du module de defense...")

    # Charger les modeles existants
    models = load_models()
    if not models:
        logger.error("Aucun modele trouve dans outputs/models/. Lancez d'abord main.py")
        sys.exit(1)

    # Charger le dataset
    csv_candidates = [
        Path("../Codes/PhishFOE_ds/PhishFOE_ds.csv"),
        Path("data/phishing.csv"),
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        logger.error("Dataset introuvable.")
        sys.exit(1)

    data = prepare_data(str(csv_path))

    results = run_defense_pipeline(
        models        = models,
        X_train       = data['X_train'],
        y_train       = data['y_train'],
        X_test        = data['X_test'],
        y_test        = data['y_test'],
        feature_names = data['feature_names'],
        epsilon       = 0.1,
        augmentation_ratio = 0.30,
    )

    logger.info("\nPipeline de defense terminee avec succes!")
    logger.info("Resultats dans: outputs/robustness/defense/")
