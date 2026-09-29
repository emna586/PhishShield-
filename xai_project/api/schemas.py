"""
Schemas Pydantic pour l'API FastAPI Phishing Detection.
Tous les modèles de requête et réponse sont définis ici.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional


class PredictionRequest(BaseModel):
    """Requête d'analyse d'URL."""
    url: str = Field(
        ...,
        description="URL complète à analyser",
        examples=["http://phishing-bank-login.com/secure/verify"]
    )

    @field_validator('url')
    @classmethod
    def url_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("L'URL ne peut pas être vide.")
        return v


class RuleDetail(BaseModel):
    """Détail d'une règle heuristique individuelle."""
    triggered:   bool  = Field(..., description="La règle a été déclenchée")
    value:       float = Field(..., description="Valeur mesurée")
    points:      int   = Field(..., description="Points attribués")
    description: str   = Field(..., description="Explication du résultat")


class HeuristicResponse(BaseModel):
    """Résultat de l'analyse heuristique."""
    score:         int                   = Field(..., description="Score heuristique cumulé")
    max_score:     int                   = Field(..., description="Score maximum possible")
    score_percent: float                 = Field(..., description="Score en pourcentage (0-100)")
    details:       Dict[str, RuleDetail] = Field(default_factory=dict, description="Détails par règle")
    features:      Dict[str, float]      = Field(default_factory=dict, description="Valeurs numériques brutes")


class FeatureContribution(BaseModel):
    """Contribution d'une feature à la prédiction (SHAP ou LIME)."""
    feature:    str   = Field(..., description="Nom de la feature")
    value:      float = Field(..., description="Valeur numérique de la feature")
    importance: float = Field(..., description="Importance absolue")
    effect:     str   = Field(..., description="'Favorise Phishing' ou 'Favorise Legitime'")


class XAIExplanation(BaseModel):
    """Explications XAI en temps réel."""
    shap: List[FeatureContribution] = Field(default_factory=list, description="Top 5 SHAP")
    lime: List[FeatureContribution] = Field(default_factory=list, description="Top 5 LIME")


class ModelRobustnessInfo(BaseModel):
    """Métriques de robustesse adversariale d'un modèle."""
    vulnerable_to_fgsm: bool  = Field(..., description="Vulnérable à FGSM (drop > 15%)")
    fgsm_drop:          float = Field(..., description="Chute de précision FGSM")
    pgd_drop:           float = Field(..., description="Chute de précision PGD")
    robustness_score:   float = Field(..., description="Score global de robustesse (0-1)")


class PredictionResponse(BaseModel):
    """Réponse complète d'analyse d'URL."""
    url:              str                             = Field(..., description="URL analysée")
    status:           str                             = Field(..., description="'phishing' ou 'legitimate'")
    confidence:       float                           = Field(..., description="Confiance du verdict (0.5-1.0)")
    method:           str                             = Field(..., description="Couche décisive (whitelist/blacklist/heuristics/hybrid)")
    score:            float                           = Field(..., description="Score combiné /100")
    verdict_details:  str                             = Field(..., description="Explication résumée du verdict")
    heuristics:       HeuristicResponse               = Field(..., description="Détails heuristiques")
    ml_probabilities: Dict[str, float]                = Field(default_factory=dict, description="Probabilités ML par modèle")
    explanations:     Optional[XAIExplanation]        = Field(None, description="Explications SHAP+LIME (si ML actif)")
    robustness:       Dict[str, ModelRobustnessInfo]  = Field(default_factory=dict, description="Métriques de robustesse par modèle")


class ListUpdateRequest(BaseModel):
    """Requête pour ajouter/supprimer un domaine des listes."""
    domain:    str = Field(..., description="Nom de domaine (ex: google.com)", examples=["bad-site.net"])
    list_type: str = Field(..., description="'whitelist' ou 'blacklist'", examples=["blacklist"])

    @field_validator('list_type')
    @classmethod
    def validate_list_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ('whitelist', 'blacklist'):
            raise ValueError("list_type doit être 'whitelist' ou 'blacklist'.")
        return v


class SimpleResponse(BaseModel):
    """Réponse simple (confirmation d'action)."""
    success: bool = Field(..., description="Succès de l'action")
    message: str  = Field(..., description="Message de statut")
