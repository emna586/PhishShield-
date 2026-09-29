"""
Module XAI Validation - Validation croisée triangulaire SHAP + LIME + PFI
"""

import numpy as np
from typing import Dict, Tuple, Set
from utils import setup_logger
from config import COHERENCE_THRESHOLDS

logger = setup_logger(__name__)


def calculate_overlap_percentage(set1: np.ndarray, set2: np.ndarray) -> float:
    """
    Calcule le pourcentage d'overlap entre deux sets de Top-10
    
    Args:
        set1: Array d'indices du premier set
        set2: Array d'indices du deuxième set
        
    Returns:
        Pourcentage d'overlap (0-100)
    """
    intersection = len(set(set1).intersection(set(set2)))
    overlap = (intersection / 10) * 100
    return overlap


def calculate_correlation(imp1: np.ndarray, imp2: np.ndarray) -> float:
    """
    Calcule la corrélation entre deux vecteurs d'importance
    
    Args:
        imp1: Vector d'importance 1
        imp2: Vector d'importance 2
        
    Returns:
        Coefficient de corrélation
    """
    # Gérer les cas edge (vecteurs constants)
    if np.std(imp1) == 0 or np.std(imp2) == 0:
        return 0.0
    
    correlation = np.corrcoef(imp1, imp2)[0, 1]
    
    # Gérer les cas NaN
    if np.isnan(correlation):
        return 0.0
    
    return correlation


def get_consensus_features(top10_shap: np.ndarray, top10_lime: np.ndarray,
                           top10_pfi: np.ndarray) -> Set[int]:
    """
    Identifie les features consensus (présentes dans les 3 Top-10)
    
    Args:
        top10_shap: Indices des top 10 features SHAP
        top10_lime: Indices des top 10 features LIME
        top10_pfi: Indices des top 10 features PFI
        
    Returns:
        Set des indices des features consensus
    """
    consensus = set(top10_shap) & set(top10_lime) & set(top10_pfi)
    return consensus


def triangular_xai_validation(shap_importance: np.ndarray, lime_importance: np.ndarray,
                              pfi_importance: np.ndarray, top10_shap: np.ndarray,
                              top10_lime: np.ndarray, top10_pfi: np.ndarray,
                              feature_names: list, model_name: str) -> Dict:
    """
    Validation croisée triangulaire SHAP + LIME + PFI
    
    Calcule:
    1. Les corrélations entre les 3 méthodes
    2. L'overlap des Top-10 features
    3. Les features consensus
    
    Args:
        shap_importance: Vector d'importance SHAP
        lime_importance: Vector d'importance LIME
        pfi_importance: Vector d'importance PFI
        top10_shap: Indices des top 10 SHAP
        top10_lime: Indices des top 10 LIME
        top10_pfi: Indices des top 10 PFI
        feature_names: Noms des features
        model_name: Nom du modèle pour logging
        
    Returns:
        Dict avec résultats de validation croisée
    """
    logger.info("=" * 70)
    logger.info(f"VALIDATION CROISÉE TRIANGULAIRE XAI - {model_name}")
    logger.info("=" * 70)
    
    # 1. Calculer les corrélations
    logger.info("\n1. Corrélations entre les 3 méthodes:")
    
    shap_lime_corr = calculate_correlation(shap_importance, lime_importance)
    shap_pfi_corr = calculate_correlation(shap_importance, pfi_importance)
    lime_pfi_corr = calculate_correlation(lime_importance, pfi_importance)
    
    logger.info(f"   SHAP ↔ LIME: {shap_lime_corr:.4f} (seuil: {COHERENCE_THRESHOLDS['shap_lime_corr']:.2f})")
    logger.info(f"   SHAP ↔ PFI:  {shap_pfi_corr:.4f} (seuil: {COHERENCE_THRESHOLDS['shap_pfi_corr']:.2f})")
    logger.info(f"   LIME ↔ PFI:  {lime_pfi_corr:.4f} (seuil: {COHERENCE_THRESHOLDS['lime_pfi_corr']:.2f})")
    
    # 2. Calculer l'overlap Top-10
    logger.info("\n2. Overlap des Top-10 features:")
    
    shap_lime_overlap = calculate_overlap_percentage(top10_shap, top10_lime)
    shap_pfi_overlap = calculate_overlap_percentage(top10_shap, top10_pfi)
    lime_pfi_overlap = calculate_overlap_percentage(top10_lime, top10_pfi)
    
    logger.info(f"   SHAP ↔ LIME: {shap_lime_overlap:.1f}% (seuil: {COHERENCE_THRESHOLDS['shap_lime_overlap']:.0f}%)")
    logger.info(f"   SHAP ↔ PFI:  {shap_pfi_overlap:.1f}% (seuil: {COHERENCE_THRESHOLDS['shap_pfi_overlap']:.0f}%)")
    logger.info(f"   LIME ↔ PFI:  {lime_pfi_overlap:.1f}% (seuil: {COHERENCE_THRESHOLDS['lime_pfi_overlap']:.0f}%)")
    
    # 3. Identifier les features consensus
    logger.info("\n3. Features Consensus:")
    
    consensus_features = get_consensus_features(top10_shap, top10_lime, top10_pfi)
    logger.info(f"   {len(consensus_features)} features consensus (présentes dans les 3 Top-10):")
    
    for feat_idx in sorted(consensus_features):
        feat_name = feature_names[feat_idx]
        logger.info(f"     • {feat_name}")
    
    # 4. Évaluer la cohérence
    logger.info("\n4. Évaluation de la Cohérence:")
    
    checks = {
        'SHAP_LIME_corr': shap_lime_corr >= COHERENCE_THRESHOLDS['shap_lime_corr'],
        'SHAP_PFI_corr': shap_pfi_corr >= COHERENCE_THRESHOLDS['shap_pfi_corr'],
        'LIME_PFI_corr': lime_pfi_corr >= COHERENCE_THRESHOLDS['lime_pfi_corr'],
        'SHAP_LIME_overlap': shap_lime_overlap >= COHERENCE_THRESHOLDS['shap_lime_overlap'],
        'SHAP_PFI_overlap': shap_pfi_overlap >= COHERENCE_THRESHOLDS['shap_pfi_overlap'],
        'LIME_PFI_overlap': lime_pfi_overlap >= COHERENCE_THRESHOLDS['lime_pfi_overlap']
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check_name, passed_check in checks.items():
        status = "✓ PASS" if passed_check else "✗ FAIL"
        logger.info(f"   {status}: {check_name}")
    
    logger.info(f"\n   Résultat Global: {passed}/{total} vérifications validées ({passed/total*100:.0f}%)")
    
    coherence_status = "COHÉRENT" if passed >= 5 else "INCOHÉRENT"
    logger.info(f"   Conclusion: {coherence_status}")
    
    logger.info("=" * 70)
    
    # Retourner les résultats
    return {
        'model_name': model_name,
        'correlations': {
            'SHAP_LIME_corr': shap_lime_corr,
            'SHAP_PFI_corr': shap_pfi_corr,
            'LIME_PFI_corr': lime_pfi_corr
        },
        'overlaps': {
            'SHAP_LIME_overlap': shap_lime_overlap,
            'SHAP_PFI_overlap': shap_pfi_overlap,
            'LIME_PFI_overlap': lime_pfi_overlap
        },
        'consensus_features': sorted(list(consensus_features)),
        'consensus_features_names': [feature_names[i] for i in sorted(consensus_features)],
        'checks': checks,
        'passed': passed,
        'total': total,
        'coherent': passed >= 5
    }
