"""
Module de règles heuristiques pour la détection de phishing
"""

from urllib.parse import urlparse
from typing import Dict, Any
from utils import setup_logger

logger = setup_logger(__name__)


class HeuristicEngine:
    """
    Moteur de règles heuristiques statiques évaluant la dangerosité d'une URL
    en fonction de critères structurels (HTTPS, longueur, sous-domaines, points,
    caractères d'obfuscation et chiffres).
    Poids maximum cumulé de 85 points.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialise le moteur heuristique avec les configurations de règles par défaut.
        
        Args:
            config: Optionnel. Dictionnaire pour écraser la configuration par défaut.
        """
        default_config = {
            'rules': {
                'IsHTTPS': {
                    'weight': 10,
                    'description': "Pas de protocole sécurisé HTTPS (HTTP brut)",
                },
                'URLLength': {
                    'weight': 15,
                    'threshold': 75,
                    'description': "Longueur de l'URL excessive",
                },
                'NoOfSubDomain': {
                    'weight': 15,
                    'threshold': 3,
                    'description': "Nombre élevé de sous-domaines",
                },
                'NoOfDots': {
                    'weight': 10,
                    'threshold': 3,
                    'description': "Nombre élevé de points dans l'URL",
                },
                'NoOfObfuscatedChar': {
                    'weight': 10,
                    'threshold': 1,
                    'description': "Présence de caractères suspectés d'obfuscation (@, % ou tirets multiples)",
                },
                'NoOfDigits': {
                    'weight': 25,
                    'threshold': 5,
                    'description': "Nombre élevé de chiffres dans l'URL",
                }
            }
        }
        
        self.config = default_config
        if config:
            # Fusionner les configurations
            if 'rules' in config:
                for key, val in config['rules'].items():
                    if key in self.config['rules']:
                        self.config['rules'][key].update(val)

    def get_features(self, url: str) -> Dict[str, int]:
        """
        Analyse une URL brute et extrait les 6 métriques nécessaires pour les heuristiques.
        
        Args:
            url: URL à analyser.
            
        Returns:
            Dictionnaire contenant les valeurs extraites.
        """
        url = url.strip()
        is_https = 1 if url.lower().startswith("https://") else 0
        
        # Normaliser pour urlparse
        temp_url = url
        if not temp_url.startswith(("http://", "https://")):
            temp_url = "http://" + temp_url
            
        try:
            parsed = urlparse(temp_url)
            domain = parsed.netloc.lower()
            if ":" in domain:
                domain = domain.split(":")[0]
        except Exception:
            domain = ""
            
        # 1. IsHTTPS (déjà fait)
        # 2. URLLength
        url_len = len(url)
        
        # 3. NoOfSubDomain (compter les composants du domaine)
        parts = [p for p in domain.split('.') if p]
        if len(parts) <= 2:
            no_subdomain = 0
        else:
            # Gérer les TLDs à deux niveaux (ex: .co.uk, .com.ar, .asso.fr)
            tld_two_levels = {'com', 'co', 'ac', 'org', 'net', 'gov', 'edu', 'asso'}
            if len(parts[-1]) == 2 and parts[-2] in tld_two_levels and len(parts) >= 3:
                no_subdomain = max(0, len(parts) - 3)
            else:
                no_subdomain = max(0, len(parts) - 2)
                
        # 4. NoOfDots
        no_dots = url.count('.')
        
        # 5. NoOfObfuscatedChar (nombre de @, % et tirets dans le domaine)
        no_obfuscated = url.count('@') + url.count('%') + domain.count('-')
        
        # 6. NoOfDigits
        no_digits = sum(c.isdigit() for c in url)
        
        return {
            'IsHTTPS': is_https,
            'URLLength': url_len,
            'NoOfSubDomain': no_subdomain,
            'NoOfDots': no_dots,
            'NoOfObfuscatedChar': no_obfuscated,
            'NoOfDigits': no_digits
        }

    def evaluate(self, url: str) -> Dict[str, Any]:
        """
        Évalue une URL par rapport aux 6 règles d'heuristique et compile le score.
        
        Args:
            url: URL à évaluer.
            
        Returns:
            Un dictionnaire contenant le score global, le score en pourcentage, 
            les détails par règle et les features extraites.
        """
        features = self.get_features(url)
        rules = self.config['rules']
        
        score = 0
        max_possible_score = sum(r['weight'] for r in rules.values())
        details = {}
        
        # Règle 1: IsHTTPS
        rule_cfg = rules['IsHTTPS']
        if features['IsHTTPS'] == 0:
            score += rule_cfg['weight']
            details['IsHTTPS'] = {
                'triggered': True,
                'value': 0,
                'points': rule_cfg['weight'],
                'description': rule_cfg['description']
            }
        else:
            details['IsHTTPS'] = {
                'triggered': False,
                'value': 1,
                'points': 0,
                'description': "URL sécurisée par protocole HTTPS"
            }
            
        # Règle 2: URLLength
        rule_cfg = rules['URLLength']
        if features['URLLength'] > rule_cfg['threshold']:
            score += rule_cfg['weight']
            details['URLLength'] = {
                'triggered': True,
                'value': features['URLLength'],
                'points': rule_cfg['weight'],
                'description': f"{rule_cfg['description']} ({features['URLLength']} > {rule_cfg['threshold']} caractères)"
            }
        else:
            details['URLLength'] = {
                'triggered': False,
                'value': features['URLLength'],
                'points': 0,
                'description': "Longueur de l'URL normale"
            }
            
        # Règle 3: NoOfSubDomain
        rule_cfg = rules['NoOfSubDomain']
        if features['NoOfSubDomain'] >= rule_cfg['threshold']:
            score += rule_cfg['weight']
            details['NoOfSubDomain'] = {
                'triggered': True,
                'value': features['NoOfSubDomain'],
                'points': rule_cfg['weight'],
                'description': f"{rule_cfg['description']} ({features['NoOfSubDomain']} >= {rule_cfg['threshold']} sous-domaines)"
            }
        else:
            details['NoOfSubDomain'] = {
                'triggered': False,
                'value': features['NoOfSubDomain'],
                'points': 0,
                'description': "Nombre de sous-domaines normal"
            }
            
        # Règle 4: NoOfDots
        rule_cfg = rules['NoOfDots']
        if features['NoOfDots'] >= rule_cfg['threshold']:
            score += rule_cfg['weight']
            details['NoOfDots'] = {
                'triggered': True,
                'value': features['NoOfDots'],
                'points': rule_cfg['weight'],
                'description': f"{rule_cfg['description']} ({features['NoOfDots']} >= {rule_cfg['threshold']} points)"
            }
        else:
            details['NoOfDots'] = {
                'triggered': False,
                'value': features['NoOfDots'],
                'points': 0,
                'description': "Nombre de points dans l'URL normal"
            }
            
        # Règle 5: NoOfObfuscatedChar
        rule_cfg = rules['NoOfObfuscatedChar']
        if features['NoOfObfuscatedChar'] >= rule_cfg['threshold']:
            score += rule_cfg['weight']
            details['NoOfObfuscatedChar'] = {
                'triggered': True,
                'value': features['NoOfObfuscatedChar'],
                'points': rule_cfg['weight'],
                'description': f"{rule_cfg['description']} ({features['NoOfObfuscatedChar']} indicateur(s) trouvé(s))"
            }
        else:
            details['NoOfObfuscatedChar'] = {
                'triggered': False,
                'value': features['NoOfObfuscatedChar'],
                'points': 0,
                'description': "Aucun caractère suspect détecté"
            }
            
        # Règle 6: NoOfDigits
        rule_cfg = rules['NoOfDigits']
        if features['NoOfDigits'] >= rule_cfg['threshold']:
            score += rule_cfg['weight']
            details['NoOfDigits'] = {
                'triggered': True,
                'value': features['NoOfDigits'],
                'points': rule_cfg['weight'],
                'description': f"{rule_cfg['description']} ({features['NoOfDigits']} >= {rule_cfg['threshold']} chiffres)"
            }
        else:
            details['NoOfDigits'] = {
                'triggered': False,
                'value': features['NoOfDigits'],
                'points': 0,
                'description': "Nombre de chiffres dans l'URL normal"
            }
            
        score_percent = (score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        return {
            'score': score,
            'max_score': max_possible_score,
            'score_percent': score_percent,
            'details': details,
            'features': features
        }
