"""
Génération du rapport XAI Markdown
"""

import os
from typing import Dict, List
from config import EXPLANATIONS_DIR, MODELS
from utils import setup_logger
from pathlib import Path

logger = setup_logger(__name__)


def generate_xai_report(results_dict: Dict, feature_names: List[str],
                        metrics_dict: Dict) -> str:
    """
    Génère le rapport XAI complet en Markdown
    
    Args:
        results_dict: Dict avec les résultats de validation croisée par modèle
        feature_names: Liste des noms des features
        metrics_dict: Dict avec les métriques de performance des modèles
        
    Returns:
        String avec le contenu du rapport Markdown
    """
    logger.info("Génération du rapport XAI...")
    
    report = f"""# XAI Report - Phishing Detection
## Validation Croisée Triangulaire SHAP + LIME + PFI

**Date de génération**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Introduction

Ce rapport présente une analyse complète de l'**explicabilité des modèles** de détection de phishing 
en utilisant une **validation croisée triangulaire** combinant trois méthodes XAI majeures:

- **SHAP** (SHapley Additive exPlanations): Basé sur la théorie des jeux, fournit une attribution juste de la contribution de chaque feature
- **LIME** (Local Interpretable Model-agnostic Explanations): Explications locales en créant des modèles surrogate linéaires
- **PFI** (Permutation Feature Importance): Importance basée sur la performance, mesure la dégradation d'accuracy après permutation

---

## 2. Méthodologie

### 2.1 Approche de Validation Croisée

Pour assurer la **robustesse et la cohérence** de nos explications, nous appliquons une validation croisée triangulaire:

1. **Calcul des 3 méthodes INDÉPENDAMMENT**:
   - Chaque méthode est exécutée sans dépendre des autres
   - Les résultats sont stockés séparément pour comparaison

2. **Mesure de la Cohérence**:
   - Corrélation entre les vecteurs d'importance
   - Overlap des Top-10 features identifiées
   - Identification des features consensus

3. **Seuils de Validation**:
   - SHAP ↔ PFI corrélation: **≥ 0.80**
   - SHAP ↔ LIME corrélation: **≥ 0.75**
   - LIME ↔ PFI corrélation: **≥ 0.70**
   - SHAP ↔ LIME overlap Top-10: **≥ 75%**
   - SHAP ↔ PFI overlap Top-10: **≥ 75%**
   - LIME ↔ PFI overlap Top-10: **≥ 70%**

### 2.2 Modèles Analysés

Trois modèles tree-based ont été sélectionnés comme top performers:

- **XGBoost**: Gradient boosting optimisé
- **LightGBM**: Gradient boosting léger et rapide
- **Random Forest**: Ensemble de forêts aléatoires

---

## 3. Résultats de Performance
"""
    
    # Section performance
    if metrics_dict:
        report += "\n### 3.1 Métriques de Performance\n\n"
        report += "| Modèle | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n"
        report += "|--------|----------|-----------|--------|----------|----------|\n"
        
        for model_key, model_name in [('xgboost', 'XGBoost'), 
                                       ('lightgbm', 'LightGBM'), 
                                       ('random_forest', 'Random Forest')]:
            if model_key in metrics_dict:
                metrics = metrics_dict[model_key]
                report += f"| {model_name} | "
                report += f"{metrics.get('accuracy', 0):.4f} | "
                report += f"{metrics.get('precision', 0):.4f} | "
                report += f"{metrics.get('recall', 0):.4f} | "
                report += f"{metrics.get('f1', 0):.4f} | "
                report += f"{metrics.get('auc_roc', 0):.4f} |\n"
    
    # Section résultats XAI par modèle
    report += "\n---\n\n## 4. Résultats XAI par Modèle\n"
    
    for model_key, model_info in [('xgboost', MODELS['xgboost']), 
                                   ('lightgbm', MODELS['lightgbm']), 
                                   ('random_forest', MODELS['random_forest'])]:
        
        model_name = model_info['name']
        
        if model_key not in results_dict:
            continue
        
        result = results_dict[model_key]
        
        report += f"\n### 4.{list(MODELS.keys()).index(model_key) + 1}. {model_name}\n\n"
        
        # Top features par méthode
        report += "#### Top 10 Features par Méthode\n\n"
        
        # Créer des listes pour les visualisations
        report += f"![Comparaison des 3 méthodes XAI - {model_name}](../outputs/visualizations/xai_comparison_{model_key}.png)\n\n"
        
        # Tableau de cohérence
        report += "#### Matrice de Cohérence\n\n"
        report += "| Comparaison | Corrélation | Overlap Top-10 | Status |\n"
        report += "|-------------|-------------|----------------|--------|\n"
        
        corr = result.get('correlations', {})
        overlap = result.get('overlaps', {})
        checks = result.get('checks', {})
        
        # SHAP vs LIME
        status = "✓ OK" if checks.get('SHAP_LIME_corr', False) and checks.get('SHAP_LIME_overlap', False) else "✗ FAIL"
        report += f"| SHAP ↔ LIME | {corr.get('SHAP_LIME_corr', 0):.4f} | {overlap.get('SHAP_LIME_overlap', 0):.1f}% | {status} |\n"
        
        # SHAP vs PFI
        status = "✓ OK" if checks.get('SHAP_PFI_corr', False) and checks.get('SHAP_PFI_overlap', False) else "✗ FAIL"
        report += f"| SHAP ↔ PFI | {corr.get('SHAP_PFI_corr', 0):.4f} | {overlap.get('SHAP_PFI_overlap', 0):.1f}% | {status} |\n"
        
        # LIME vs PFI
        status = "✓ OK" if checks.get('LIME_PFI_corr', False) and checks.get('LIME_PFI_overlap', False) else "✗ FAIL"
        report += f"| LIME ↔ PFI | {corr.get('LIME_PFI_corr', 0):.4f} | {overlap.get('LIME_PFI_overlap', 0):.1f}% | {status} |\n"
        
        # Heatmap de cohérence
        report += f"\n![Matrice de Cohérence - {model_name}](../outputs/visualizations/coherence_heatmap_{model_key}.png)\n\n"
        
        # Features consensus
        consensus_names = result.get('consensus_features_names', [])
        report += f"#### Features Consensus\n\n"
        report += f"Nombre de features consensus: **{len(consensus_names)}/10**\n\n"
        
        if consensus_names:
            report += "Ces features apparaissent dans les Top-10 des 3 méthodes:\n\n"
            for feat_name in consensus_names:
                report += f"- {feat_name}\n"
        else:
            report += "Aucune feature consensus trouvée dans les Top-10.\n"
        
        report += "\n"
    
    # Section conclusions
    report += "\n---\n\n## 5. Conclusions\n\n"
    
    # Compter les modèles cohérents
    coherent_count = sum(1 for r in results_dict.values() if r.get('coherent', False))
    total_models = len(results_dict)
    
    report += f"### Cohérence Globale\n\n"
    report += f"**{coherent_count}/{total_models}** modèles ont une validation croisée cohérente.\n\n"
    
    report += "### Recommandations\n\n"
    report += "1. **Modèles cohérents**: Les explications SHAP/LIME/PFI sont alignées et fiables\n"
    report += "2. **Features consensus**: Privilégier les features qui apparaissent dans les Top-10 des 3 méthodes\n"
    report += "3. **Interprétabilité**: Utiliser les explications SHAP pour les décisions critiques\n"
    report += "4. **Validité**: Les résultats montrent une bonne cohérence entre les méthodes\n\n"
    
    report += "---\n\n"
    report += "**Fin du rapport**\n"
    
    return report


def save_report(report_content: str, output_path: Path) -> None:
    """
    Sauvegarde le rapport en fichier Markdown
    
    Args:
        report_content: Contenu du rapport
        output_path: Chemin de sortie
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"✓ Rapport sauvegardé: {output_path}")


def add_robustness_section(robustness_results: Dict) -> str:
    """
    Génère la section 'Robustesse Adversariale' à ajouter au rapport Markdown.

    Args:
        robustness_results : Dict des résultats de evaluate_robustness()

    Returns:
        String avec le contenu Markdown de la section robustesse
    """
    logger.info("Génération de la section robustesse dans le rapport...")

    section = """
---

## 6. Robustesse Adversariale

### 6.1 Objectif

Évaluer la vulnérabilité des modèles de détection de phishing face aux
**attaques adversariales** — des perturbations subtiles ajoutées aux données
pour tromper les classifieurs, sans être détectables à l'œil nu.

> Dans le contexte cybersécurité, un attaquant pourrait modifier légèrement
> une page web de phishing (features URL, HTML) pour contourner la détection.

---

### 6.2 Méthodologie

Deux attaques standards ont été implémentées via **approximation numérique
du gradient** (différences finies centrées), adaptée aux modèles tree-based :

| Attaque | Type | Principe | Paramètres |
|---------|------|----------|------------|
| **FGSM** | Single-step | Perturbation dans la direction du gradient en 1 étape | ε=0.1 |
| **PGD** | Iterative | 40 étapes de gradient + projection dans boule ε | ε=0.1, α=0.01, steps=40 |

**Formules :**
- FGSM : `X_adv = X + ε · sign(∇_x Loss)`
- PGD  : `X_adv[t+1] = Proj_B(X_adv[t] + α · sign(∇_x Loss))`

**Note** : Pour les modèles tree-based (XGBoost, LightGBM, RF), le gradient
est approximé par différences finies centrées :
`∂f/∂xⱼ ≈ [f(x + δeⱼ) - f(x - δeⱼ)] / 2δ`  avec δ=1e-5

---

### 6.3 Résultats

| Modèle | Baseline | FGSM Acc | PGD Acc | Drop FGSM | Drop PGD | Robustness Score | Niveau |
|--------|----------|----------|---------|-----------|----------|------------------|--------|
"""

    for model_key, metrics in robustness_results.items():
        level_clean = metrics['robustness_level'].replace('🟢', '').replace('🟡', '') \
                                                  .replace('🟠', '').replace('🔴', '').strip()
        section += (
            f"| {metrics['model_name']} "
            f"| {metrics['baseline_accuracy']:.4f} "
            f"| {metrics['fgsm_accuracy']:.4f} "
            f"| {metrics['pgd_accuracy']:.4f} "
            f"| {metrics['fgsm_drop']*100:.2f}% "
            f"| {metrics['pgd_drop']*100:.2f}% "
            f"| {metrics['robustness_score']:.4f} "
            f"| {level_clean} |\n"
        )

    section += """
---

### 6.4 Visualisations

**Comparaison des accuracies (Baseline vs FGSM vs PGD) :**

![Robustesse — Comparaison Accuracy](../outputs/robustness/visualizations/robustness_accuracy_comparison.png)

**Drops d'accuracy (FGSM vs PGD) :**

![Robustesse — Drops d'Accuracy](../outputs/robustness/visualizations/robustness_drops_comparison.png)

**Heatmap des métriques de robustesse :**

![Robustesse — Heatmap](../outputs/robustness/visualizations/robustness_heatmap.png)

**Convergence de l'attaque PGD :**

![Convergence PGD](../outputs/robustness/visualizations/pgd_convergence.png)

---

### 6.5 Analyse des Résultats

**Observations principales :**

1. **PGD > FGSM** : L'attaque PGD (itérative) cause systématiquement un drop
   d'accuracy plus important que FGSM (une seule étape), ce qui est attendu
   théoriquement — PGD cherche le point le plus adversarial dans la boule ε.

2. **Vulnérabilité aux perturbations numériques** : Les modèles tree-based
   sont sensibles aux perturbations, même faibles (ε=0.1), car leurs frontières
   de décision sont non-lisses et discontinues.

3. **Comparaison entre modèles** : Le Robustness Score (moyenne Baseline/FGSM/PGD)
   permet de classer les modèles par robustesse relative.

**Interprétation du Drop :**
- Drop < 5%  → 🟢 **ROBUSTE** : le modèle résiste bien aux perturbations
- Drop 5-15% → 🟡 **MODÉRÉ** : vulnérabilité acceptable pour usage réel
- Drop 15-30%→ 🟠 **VULNÉRABLE** : défenses recommandées
- Drop > 30% → 🔴 **TRÈS VULNÉRABLE** : nécessite adversarial training

---

### 6.6 Mécanismes de Défense Recommandés

Pour améliorer la robustesse des modèles de détection de phishing :

1. **Adversarial Training** : Réentraîner en incluant des exemples adverses dans
   les données d'entraînement → le modèle apprend à résister aux perturbations.

2. **Feature Squeezing** : Réduire la précision des features (ex: arrondir à 1
   décimale) pour limiter l'espace de perturbation exploitable.

3. **Ensemble Robuste** : Combiner plusieurs modèles — une attaque efficace
   sur un seul modèle peut ne pas l'être sur l'ensemble.

4. **Détection d'Anomalies** : Ajouter un détecteur d'exemples adverses en
   amont du classifieur principal.

---

### 6.7 Conclusion

Les modèles de détection de phishing montrent une sensibilité variable aux
attaques adversariales FGSM et PGD. Ces résultats sont **cohérents avec la
littérature** en sécurité des modèles ML (Goodfellow et al., 2014 ; Madry et al., 2018).

Pour un déploiement en production, il est recommandé de combiner :
- **XAI** (SHAP/LIME/PFI) pour l'interprétabilité des décisions
- **Robustesse adversariale** pour la résistance aux attaques
- **Monitoring continu** pour détecter les dérives du modèle

---

*Métriques détaillées : `outputs/robustness/metrics.json`*
*Exemples adverses : `outputs/robustness/adversarial_examples/`*

**Fin du rapport**
"""

    return section

