"""
xai_deep_learning.py — Explainable AI for Deep Learning Models
===============================================================
Applies SHAP GradientExplainer + LIME on MLP, CNN, LSTM models.
Generates feature importance visualizations and a comparison
with ML XAI results.

Usage:
    cd xai_project
    python xai_deep_learning.py
"""

import sys
import json
import logging
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

# Fix UTF-8 Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR  = ROOT / "outputs" / "models"
DL_DIR      = MODELS_DIR / "deep_learning"
VIZ_DIR     = ROOT / "outputs" / "visualizations"
XAI_DL_DIR  = ROOT / "outputs" / "xai_deep_learning"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
XAI_DL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    from utils.data_loader import load_phishing_dataset
    from sklearn.model_selection import train_test_split

    possible_paths = [
        ROOT.parent / "Codes" / "PhishFOE_ds" / "PhishFOE_ds.csv",
        ROOT / "data" / "phishing.csv",
        ROOT.parent / "Codes" / "Phishing_Legitimate_full_codes" / "Phishing_Legitimate_full.csv",
    ]
    dataset_path = next((p for p in possible_paths if p.exists()), None)
    if dataset_path is None:
        raise FileNotFoundError("Dataset introuvable. Verifiez le dossier Codes/.")

    logger.info(f"Dataset: {dataset_path}")
    df_raw, X, y, feature_names = load_phishing_dataset(str(dataset_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply scaler (same one used for ML models)
    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler  = joblib.load(scaler_path)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)
        logger.info("StandardScaler applique.")

    # Convert to numpy
    if hasattr(X_train, "values"): X_train = X_train.values
    if hasattr(X_test,  "values"): X_test  = X_test.values
    if hasattr(y_train, "values"): y_train = y_train.values
    if hasattr(y_test,  "values"): y_test  = y_test.values

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Features ({len(feature_names)}): {feature_names[:5]}...")
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DL MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_dl_models():
    import tensorflow as tf

    models = {}
    model_files = {
        "MLP":  DL_DIR / "mlp_model.keras",
        "CNN":  DL_DIR / "cnn_model.keras",
        "LSTM": DL_DIR / "lstm_model.keras",
    }

    for name, path in model_files.items():
        if not path.exists():
            logger.warning(f"[WARN] Modele {name} introuvable: {path}")
            logger.warning(f"       Lancez d'abord: python train_deep_learning.py")
            continue
        try:
            model = tf.keras.models.load_model(str(path))
            models[name] = model
            logger.info(f"[OK] Modele {name} charge depuis {path}")
        except Exception as e:
            logger.error(f"[ERR] Chargement {name}: {e}")

    if not models:
        raise RuntimeError(
            "Aucun modele DL trouve. Lancez d'abord: python train_deep_learning.py"
        )

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP — GradientExplainer for Keras Models
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_dl(model, model_name, X_train, X_test, feature_names,
                    background_size=100, explain_size=200, reshape=False):
    """
    Applies SHAP GradientExplainer on a Keras model.

    Args:
        model        : Keras model (MLP, CNN, LSTM)
        model_name   : Name for logging
        X_train      : Scaled training data (2D)
        X_test       : Scaled test data (2D)
        feature_names: List of feature names
        background_size: Number of background samples for SHAP
        explain_size : Number of instances to explain
        reshape      : If True, reshape to (n, features, 1) for CNN/LSTM

    Returns:
        shap_importance: Mean absolute SHAP values [n_features]
        top10_indices  : Indices of top 10 features
        shap_values    : Full SHAP array [n_samples, n_features]
    """
    import shap
    import tensorflow as tf

    logger.info(f"\n{'─'*60}")
    logger.info(f"  SHAP GradientExplainer — {model_name}")
    logger.info(f"{'─'*60}")

    # Sample background data (representative subset of training set)
    bg_idx = np.random.choice(len(X_train), background_size, replace=False)
    X_bg   = X_train[bg_idx]

    # Sample test instances to explain
    test_idx = np.random.choice(len(X_test), min(explain_size, len(X_test)), replace=False)
    X_exp    = X_test[test_idx]

    # Reshape for CNN/LSTM: (n_samples, n_features) → (n_samples, n_features, 1)
    if reshape:
        X_bg  = X_bg.reshape(X_bg.shape[0], X_bg.shape[1], 1)
        X_exp = X_exp.reshape(X_exp.shape[0], X_exp.shape[1], 1)
        logger.info(f"  Input shape (reshaped): {X_exp.shape}")
    else:
        logger.info(f"  Input shape: {X_exp.shape}")

    logger.info(f"  Background: {X_bg.shape[0]} samples | Explain: {X_exp.shape[0]} samples")

    try:
        # GradientExplainer uses backpropagation — works with any TF/Keras model
        explainer   = shap.GradientExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_exp)

        logger.info(f"  [OK] SHAP values computed. Raw type: {type(shap_values)}")

        # Handle different output formats from GradientExplainer
        if isinstance(shap_values, list):
            # Multi-output: take the positive class (phishing = 1)
            arr = np.array(shap_values[0])
        else:
            arr = np.array(shap_values)

        logger.info(f"  SHAP array shape (before flatten): {arr.shape}")

        # Handle shapes:
        # MLP  → (n_samples, n_features)
        # CNN  → (n_samples, n_features, 1)
        # LSTM → (n_samples, n_features, 1)
        if arr.ndim == 3:
            # Squeeze last dim: (n, features, 1) → (n, features)
            arr = arr[:, :, 0]

        logger.info(f"  SHAP array shape (final): {arr.shape}")

        # Global importance: mean absolute SHAP values across all samples
        shap_importance = np.abs(arr).mean(axis=0)

        # Ensure shape matches feature count
        assert len(shap_importance) == len(feature_names), \
            f"Shape mismatch: {len(shap_importance)} SHAP values vs {len(feature_names)} features"

        top10_indices = np.argsort(shap_importance)[::-1][:10]

        logger.info(f"  Top 10 features ({model_name}):")
        for rank, idx in enumerate(top10_indices, 1):
            logger.info(f"    {rank:2d}. {feature_names[idx]:<30} SHAP={shap_importance[idx]:.4f}")

        return shap_importance, top10_indices, arr

    except Exception as e:
        logger.error(f"  [ERR] SHAP GradientExplainer failed for {model_name}: {e}")
        logger.error(f"       Falling back to zero importance vector.")
        shap_importance = np.zeros(len(feature_names))
        top10_indices   = np.arange(10)
        return shap_importance, top10_indices, None


# ─────────────────────────────────────────────────────────────────────────────
# 4. LIME — Model-agnostic, works with any predict function
# ─────────────────────────────────────────────────────────────────────────────

def compute_lime_dl(model, model_name, X_train, X_test, feature_names,
                    num_instances=50, reshape=False):
    """
    Applies LIME on a Keras model by wrapping the predict function.

    Args:
        model         : Keras model
        model_name    : Name for logging
        X_train       : Training data (2D, for background)
        X_test        : Test data (2D)
        feature_names : Feature names
        num_instances : Number of instances to explain locally
        reshape       : If True, CNN/LSTM need reshape

    Returns:
        lime_importance: Mean absolute LIME values [n_features]
        top10_indices  : Indices of top 10 features
    """
    from lime.lime_tabular import LimeTabularExplainer

    logger.info(f"\n{'─'*60}")
    logger.info(f"  LIME TabularExplainer — {model_name}")
    logger.info(f"{'─'*60}")

    # Wrap Keras predict to return 2-column probability array [P(legit), P(phishing)]
    # LIME requires predict_proba style: output shape (n_samples, n_classes)
    def predict_proba_wrapper(X_2d):
        """Converts 2D input to model format and returns 2-class probabilities."""
        X_input = X_2d
        if reshape:
            X_input = X_2d.reshape(X_2d.shape[0], X_2d.shape[1], 1)
        # Model outputs sigmoid → probability of phishing
        p_phishing = model.predict(X_input, batch_size=256, verbose=0).flatten()
        p_legit    = 1.0 - p_phishing
        return np.column_stack([p_legit, p_phishing])

    # Create LIME explainer using 2D training data as background
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=["Legitimate", "Phishing"],
            mode="classification",
            random_state=42
        )
        logger.info(f"  [OK] LimeTabularExplainer created")
    except Exception as e:
        logger.error(f"  [ERR] LIME init failed: {e}")
        return np.zeros(len(feature_names)), np.arange(10)

    # Sample test instances
    num_actual = min(num_instances, len(X_test))
    indices    = np.random.choice(len(X_test), num_actual, replace=False)
    logger.info(f"  Explaining {num_actual} instances...")

    lime_importances = np.zeros(len(feature_names))
    success_count    = 0

    for i, idx in enumerate(indices):
        try:
            exp = lime_explainer.explain_instance(
                X_test[idx],            # Always pass 2D instance to LIME
                predict_proba_wrapper,  # Wrapper handles reshape internally
                num_features=len(feature_names),
                num_samples=1000        # Perturbation samples for stability
            )

            # local_exp: dict {class_idx: [(feature_idx, weight), ...]}
            local_exp = exp.local_exp.get(1, [])

            for feat_idx, weight in local_exp:
                lime_importances[int(feat_idx)] += abs(float(weight))

            success_count += 1

            if (i + 1) % max(1, num_actual // 5) == 0:
                logger.info(f"    Progress: {i+1}/{num_actual}")

        except Exception as e:
            logger.warning(f"    [WARN] Instance {idx}: {e}")
            continue

    if success_count == 0:
        logger.error("  [ERR] All LIME explanations failed.")
        return np.zeros(len(feature_names)), np.arange(10)

    # Average over successful explanations
    lime_importances /= success_count

    top10_indices = np.argsort(np.abs(lime_importances))[::-1][:10]

    logger.info(f"  [OK] LIME done ({success_count}/{num_actual} instances). Top 10:")
    for rank, idx in enumerate(top10_indices, 1):
        logger.info(f"    {rank:2d}. {feature_names[idx]:<30} LIME={lime_importances[idx]:.4f}")

    return lime_importances, top10_indices


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_comparison(results, feature_names):
    """
    Plots SHAP feature importance for all 3 DL models side by side.
    """
    top_n = 15
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8), sharey=False)

    if n_models == 1:
        axes = [axes]

    colors = {"MLP": "#4C72B0", "CNN": "#DD8452", "LSTM": "#55A868"}

    for ax, (model_name, data) in zip(axes, results.items()):
        shap_imp = np.array(data["shap_importance"])
        top_idx  = np.argsort(shap_imp)[::-1][:top_n]
        top_vals = shap_imp[top_idx]
        top_feat = [feature_names[i] for i in top_idx]

        color = colors.get(model_name, "#888888")
        bars  = ax.barh(range(top_n), top_vals[::-1], color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feat[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
        ax.set_title(f"{model_name} — SHAP (GradientExplainer)", fontsize=12, fontweight="bold",
                     color=color, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("SHAP Feature Importance — Deep Learning Models\n(PhishShield — GradientExplainer)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = VIZ_DIR / "shap_dl_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  [OK] Saved: {path}")
    return path


def plot_lime_comparison(results, feature_names):
    """
    Plots LIME feature importance for all 3 DL models side by side.
    """
    top_n = 15
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8), sharey=False)

    if n_models == 1:
        axes = [axes]

    colors = {"MLP": "#C44E52", "CNN": "#8172B2", "LSTM": "#64B5CD"}

    for ax, (model_name, data) in zip(axes, results.items()):
        lime_imp = np.array(data["lime_importance"])
        top_idx  = np.argsort(np.abs(lime_imp))[::-1][:top_n]
        top_vals = np.abs(lime_imp[top_idx])
        top_feat = [feature_names[i] for i in top_idx]

        color = colors.get(model_name, "#888888")
        ax.barh(range(top_n), top_vals[::-1], color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feat[::-1], fontsize=9)
        ax.set_xlabel("Mean |LIME Coefficient|", fontsize=10)
        ax.set_title(f"{model_name} — LIME", fontsize=12, fontweight="bold",
                     color=color, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("LIME Feature Importance — Deep Learning Models\n(PhishShield — LimeTabularExplainer)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = VIZ_DIR / "lime_dl_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  [OK] Saved: {path}")
    return path


def plot_ml_vs_dl_xai(ml_shap_path, dl_results, feature_names):
    """
    Plots top-10 consensus features: ML SHAP vs DL SHAP overlay.
    """
    # Load ML SHAP results if available
    ml_shap = None
    if ml_shap_path and Path(ml_shap_path).exists():
        try:
            with open(ml_shap_path, "r") as f:
                ml_data = json.load(f)
            # Take LightGBM SHAP as reference for ML
            if "lightgbm" in ml_data:
                ml_shap = np.array(ml_data["lightgbm"]["shap_importance"])
        except Exception:
            pass

    # Compute average DL SHAP across MLP, CNN, LSTM
    dl_shap_arrays = []
    for model_name, data in dl_results.items():
        arr = np.array(data["shap_importance"])
        if arr.sum() > 0:
            dl_shap_arrays.append(arr)

    if not dl_shap_arrays:
        logger.warning("  [WARN] No DL SHAP data available for comparison plot.")
        return

    dl_shap_avg = np.mean(dl_shap_arrays, axis=0)

    # Normalize both to [0, 1] for comparison
    def normalize(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / rng if rng > 0 else arr

    dl_norm = normalize(dl_shap_avg)

    # Get top 15 features by DL importance
    top_idx   = np.argsort(dl_norm)[::-1][:15]
    top_feats = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 7))

    x     = np.arange(len(top_feats))
    width = 0.35

    dl_vals = dl_norm[top_idx]
    ax.bar(x - width/2, dl_vals, width, label="DL (Avg MLP+CNN+LSTM) — GradientExplainer",
           color="#DD8452", alpha=0.85, edgecolor="white")

    if ml_shap is not None:
        ml_norm    = normalize(ml_shap)
        ml_vals    = ml_norm[top_idx]
        ax.bar(x + width/2, ml_vals, width, label="ML (LightGBM) — TreeExplainer",
               color="#4C72B0", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Normalized SHAP Importance", fontsize=11)
    ax.set_title("XAI Comparison: ML (TreeExplainer) vs DL (GradientExplainer)\n"
                 "Top 15 Features by Deep Learning Importance",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    path = VIZ_DIR / "ml_vs_dl_xai_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  [OK] Saved: {path}")


def plot_top10_overlap(dl_results, feature_names):
    """
    Heatmap showing which features appear in the top-10 of each model and method.
    """
    # Collect all top-10 feature names
    all_top_features = set()
    labels = []
    top10_sets = {}

    for model_name, data in dl_results.items():
        for method in ["shap", "lime"]:
            key = f"{model_name}\n({method.upper()})"
            labels.append(key)
            top_idx = np.argsort(np.abs(np.array(data[f"{method}_importance"])))[::-1][:10]
            feats   = set(feature_names[i] for i in top_idx)
            top10_sets[key] = feats
            all_top_features.update(feats)

    all_top_features = sorted(all_top_features)
    matrix = np.zeros((len(labels), len(all_top_features)))

    for i, label in enumerate(labels):
        for j, feat in enumerate(all_top_features):
            if feat in top10_sets[label]:
                matrix[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(max(14, len(all_top_features) * 0.7), max(5, len(labels) * 0.6)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(all_top_features)))
    ax.set_xticklabels(all_top_features, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Top-10 Feature Overlap — DL Models (SHAP vs LIME)\n"
                 "Blue = Feature in Top-10 | White = Not in Top-10",
                 fontsize=12, fontweight="bold")

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(all_top_features)):
            if matrix[i, j] == 1:
                ax.text(j, i, "✓", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = VIZ_DIR / "dl_xai_top10_overlap.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  [OK] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(dl_results, feature_names):
    """Saves XAI results to JSON for report generation."""
    serializable = {}
    for model_name, data in dl_results.items():
        serializable[model_name] = {
            "shap_importance": data["shap_importance"].tolist(),
            "lime_importance":  data["lime_importance"].tolist(),
            "shap_top10": [feature_names[i] for i in data["shap_top10"]],
            "lime_top10":  [feature_names[i] for i in data["lime_top10"]],
        }

    out_path = XAI_DL_DIR / "xai_dl_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"  [OK] Results saved: {out_path}")
    return out_path


def print_summary_table(dl_results, feature_names):
    """Prints a clean summary table of top-5 features per model and method."""
    sep  = "=" * 90
    thin = "-" * 90
    print(f"\n{sep}")
    print("  XAI DEEP LEARNING — SUMMARY TABLE")
    print(sep)

    for model_name, data in dl_results.items():
        print(f"\n  {'─'*60}")
        print(f"  Model: {model_name}")
        print(f"  {'─'*60}")
        print(f"  {'Rank':<6} {'SHAP (GradientExplainer)':<32} {'LIME (TabularExplainer)'}")
        print(f"  {thin}")

        shap_top = np.argsort(data["shap_importance"])[::-1][:10]
        lime_top = np.argsort(np.abs(data["lime_importance"]))[::-1][:10]

        for rank in range(10):
            s_feat = feature_names[shap_top[rank]] if rank < len(shap_top) else "-"
            l_feat = feature_names[lime_top[rank]]  if rank < len(lime_top) else "-"
            s_val  = data["shap_importance"][shap_top[rank]] if rank < len(shap_top) else 0.0
            l_val  = data["lime_importance"][lime_top[rank]]  if rank < len(lime_top) else 0.0
            marker = "★" if rank == 0 else " "
            print(f"  {marker}{rank+1:<5} {s_feat:<30} ({s_val:.4f})   "
                  f"{l_feat:<30} ({abs(l_val):.4f})")

    print(f"\n{sep}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        logger.info(f"TensorFlow: {tf.__version__}")
    except ImportError:
        logger.error("TensorFlow non installe. Lancez: pip install tensorflow")
        sys.exit(1)

    try:
        import shap
        logger.info(f"SHAP: {shap.__version__}")
    except ImportError:
        logger.error("SHAP non installe. Lancez: pip install shap")
        sys.exit(1)

    try:
        import lime
    except ImportError:
        logger.error("LIME non installe. Lancez: pip install lime")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  PhishShield — XAI on Deep Learning Models")
    print("  SHAP GradientExplainer + LIME TabularExplainer")
    print("=" * 60)

    # ── Step 1: Load Data ───────────────────────────────────────────────────
    logger.info("\n[STEP 1] Loading dataset and scaler...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    feature_names = list(feature_names)

    # ── Step 2: Load DL Models ─────────────────────────────────────────────
    logger.info("\n[STEP 2] Loading saved Keras models...")
    dl_models = load_dl_models()

    if not dl_models:
        logger.error("No DL models loaded. Exiting.")
        sys.exit(1)

    # ── Step 3: Compute XAI for each model ─────────────────────────────────
    logger.info("\n[STEP 3] Computing SHAP + LIME for each DL model...")
    dl_results = {}

    model_configs = {
        "MLP":  {"reshape": False},
        "CNN":  {"reshape": True},
        "LSTM": {"reshape": True},
    }

    for model_name, cfg in model_configs.items():
        if model_name not in dl_models:
            logger.warning(f"  [SKIP] {model_name} not found.")
            continue

        model   = dl_models[model_name]
        reshape = cfg["reshape"]

        logger.info(f"\n{'='*60}")
        logger.info(f"  Processing: {model_name}")
        logger.info(f"{'='*60}")

        # SHAP
        shap_importance, shap_top10, shap_vals = compute_shap_dl(
            model, model_name, X_train, X_test, feature_names,
            background_size=100, explain_size=200, reshape=reshape
        )

        # LIME
        lime_importance, lime_top10 = compute_lime_dl(
            model, model_name, X_train, X_test, feature_names,
            num_instances=50, reshape=reshape
        )

        dl_results[model_name] = {
            "shap_importance": shap_importance,
            "lime_importance":  lime_importance,
            "shap_top10":      shap_top10,
            "lime_top10":      lime_top10,
        }

    if not dl_results:
        logger.error("No XAI results computed. Exiting.")
        sys.exit(1)

    # ── Step 4: Visualizations ─────────────────────────────────────────────
    logger.info("\n[STEP 4] Generating visualizations...")

    plot_shap_comparison(dl_results, feature_names)
    plot_lime_comparison(dl_results, feature_names)
    plot_top10_overlap(dl_results, feature_names)

    # ML vs DL SHAP comparison (optional, if ML results exist)
    ml_shap_path = ROOT / "outputs" / "xai_ml_results.json"
    plot_ml_vs_dl_xai(ml_shap_path, dl_results, feature_names)

    # ── Step 5: Save Results ───────────────────────────────────────────────
    logger.info("\n[STEP 5] Saving results...")
    save_results(dl_results, feature_names)

    # ── Step 6: Print Summary ──────────────────────────────────────────────
    print_summary_table(dl_results, feature_names)

    # ── Final Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [DONE] XAI Deep Learning — Complete!")
    print("=" * 60)
    print(f"\n  Visualizations saved to: {VIZ_DIR}")
    print(f"    → shap_dl_comparison.png")
    print(f"    → lime_dl_comparison.png")
    print(f"    → dl_xai_top10_overlap.png")
    print(f"    → ml_vs_dl_xai_comparison.png")
    print(f"\n  Results JSON: {XAI_DL_DIR / 'xai_dl_results.json'}")
    print("\n  These results can now be cited in Chapter 4 of the report.")
    print("=" * 60)
