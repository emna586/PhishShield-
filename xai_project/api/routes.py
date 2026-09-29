"""
Définition des routes de l'API FastAPI
"""

import sys
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Dict, Any

# Ajouter le chemin racine au path
sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import PredictionRequest, PredictionResponse, ListUpdateRequest, SimpleResponse
from api.prediction_service import PredictionService

router = APIRouter()

# Variable globale pour stocker l'instance du service
_prediction_service = None


def get_prediction_service() -> PredictionService:
    """
    Dépendance FastAPI pour obtenir l'instance unique de PredictionService.
    """
    global _prediction_service
    if _prediction_service is None:
        # Initialise le service avec les chemins par défaut
        _prediction_service = PredictionService()
    return _prediction_service


@router.post("/predict", response_model=PredictionResponse, summary="Analyser une URL")
def predict_url(
    request: PredictionRequest, 
    model: str = Query("xgboost", enum=["xgboost", "lightgbm", "random_forest"], description="Modèle principal à utiliser pour l'explication XAI"),
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Analyse une URL à l'aide de l'approche hybride (Listes + Heuristiques + Machine Learning).
    Si le Machine Learning est sollicité, fournit des explications SHAP et LIME locales en temps réel,
    ainsi que les métriques de robustesse du modèle.
    """
    try:
        result = service.predict_url(url=request.url, model_key=model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de l'analyse : {str(e)}")


@router.get("/lists", summary="Obtenir les Whitelist et Blacklist")
def get_lists(service: PredictionService = Depends(get_prediction_service)):
    """
    Retourne la liste complète des domaines whitelistés (100% de confiance légitime)
    et blacklistés (100% de confiance phishing).
    """
    try:
        wl = list(service.hybrid_engine.whitelist_blacklist.whitelist)
        bl = list(service.hybrid_engine.whitelist_blacklist.blacklist)
        return {
            "whitelist": sorted(wl),
            "blacklist": sorted(bl)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des listes : {str(e)}")


@router.post("/lists/add", response_model=SimpleResponse, summary="Ajouter un domaine aux listes")
def add_to_list(request: ListUpdateRequest, service: PredictionService = Depends(get_prediction_service)):
    """
    Ajoute un nom de domaine à la whitelist ou la blacklist.
    Le domaine est automatiquement retiré de l'autre liste s'il y était présent pour éviter les conflits.
    """
    try:
        domain = request.domain.strip()
        list_type = request.list_type.lower().strip()
        
        if list_type == "whitelist":
            success = service.hybrid_engine.whitelist_blacklist.add_to_whitelist(domain)
            msg = f"Domaine '{domain}' ajouté à la Whitelist avec succès."
        elif list_type == "blacklist":
            success = service.hybrid_engine.whitelist_blacklist.add_to_blacklist(domain)
            msg = f"Domaine '{domain}' ajouté à la Blacklist avec succès."
        else:
            raise HTTPException(status_code=400, detail="Type de liste invalide. Utilisez 'whitelist' ou 'blacklist'.")
            
        return SimpleResponse(success=success, message=msg)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'ajout : {str(e)}")


@router.post("/lists/remove", response_model=SimpleResponse, summary="Retirer un domaine des listes")
def remove_from_list(request: ListUpdateRequest, service: PredictionService = Depends(get_prediction_service)):
    """
    Retire un nom de domaine de la whitelist ou la blacklist.
    """
    try:
        domain = request.domain.strip()
        list_type = request.list_type.lower().strip()
        
        if list_type == "whitelist":
            success = service.hybrid_engine.whitelist_blacklist.remove_from_whitelist(domain)
            msg = f"Domaine '{domain}' retiré de la Whitelist." if success else f"Domaine '{domain}' non trouvé dans la Whitelist."
        elif list_type == "blacklist":
            success = service.hybrid_engine.whitelist_blacklist.remove_from_blacklist(domain)
            msg = f"Domaine '{domain}' retiré de la Blacklist." if success else f"Domaine '{domain}' non trouvé dans la Blacklist."
        else:
            raise HTTPException(status_code=400, detail="Type de liste invalide. Utilisez 'whitelist' ou 'blacklist'.")
            
        return SimpleResponse(success=success, message=msg)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du retrait : {str(e)}")
