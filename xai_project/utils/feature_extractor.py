"""
Extracteur de caractéristiques (features) depuis une URL et son contenu HTML
"""

import re
import urllib.request
from urllib.parse import urlparse
from typing import Dict, Any
from utils import setup_logger

logger = setup_logger(__name__)


def extract_features_from_url(url: str, timeout_seconds: float = 2.0) -> Dict[str, Any]:
    """
    Extrait les 29 caractéristiques numériques d'une URL pour l'alimentation des modèles ML.
    Tente de récupérer le contenu HTML de la page de manière asynchrone/synchrone rapide.
    En cas d'échec de connexion, les caractéristiques HTML sont estimées avec des valeurs par défaut.
    
    Args:
        url: L'URL à analyser.
        timeout_seconds: Délai d'attente max pour la requête HTTP.
        
    Returns:
        Dictionnaire avec les 29 caractéristiques.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url  # Par défaut https
        
    features = {}
    
    # --- 1. Caractéristiques de l'URL (10 features) ---
    features['IsHTTPS'] = 1 if url.lower().startswith("https://") else 0
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if ":" in domain:
            domain = domain.split(":")[0]
    except Exception:
        domain = ""
        
    # IsDomainIP
    ip_pattern = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')
    features['IsDomainIP'] = 1 if ip_pattern.match(domain) else 0
    
    # URLLength
    features['URLLength'] = len(url)
    
    # NoOfSubDomain
    parts = [p for p in domain.split('.') if p]
    if len(parts) <= 2:
        no_subdomain = 0
    else:
        tld_two_levels = {'com', 'co', 'ac', 'org', 'net', 'gov', 'edu', 'asso'}
        if len(parts[-1]) == 2 and parts[-2] in tld_two_levels and len(parts) >= 3:
            no_subdomain = max(0, len(parts) - 3)
        else:
            no_subdomain = max(0, len(parts) - 2)
    features['NoOfSubDomain'] = no_subdomain
    
    # NoOfDots
    features['NoOfDots'] = url.count('.')
    
    # NoOfObfuscatedChar (nombre de @, % et tirets dans le domaine)
    features['NoOfObfuscatedChar'] = url.count('@') + url.count('%') + domain.count('-')
    
    # NoOfEqual
    features['NoOfEqual'] = url.count('=')
    
    # NoOfQmark
    features['NoOfQmark'] = url.count('?')
    
    # NoOfAmp
    features['NoOfAmp'] = url.count('&')
    
    # NoOfDigits
    features['NoOfDigits'] = sum(c.isdigit() for c in url)
    
    # --- 2. Caractéristiques HTML (19 features) ---
    # Valeurs par défaut en cas d'échec de chargement
    html_features = {
        'LineLength': 0,
        'HasTitle': 0,
        'HasMeta': 0,
        'HasFavicon': 0,
        'HasExternalFormSubmit': 0,
        'HasCopyright': 0,
        'HasSocialNetworking': 0,
        'HasPasswordField': 0,
        'HasSubmitButton': 0,
        'HasKeywordBank': 0,
        'HasKeywordPay': 0,
        'HasKeywordCrypto': 0,
        'NoOfPopup': 0,
        'NoOfiFrame': 0,
        'NoOfImage': 0,
        'NoOfJS': 0,
        'NoOfCSS': 0,
        'NoOfURLRedirect': 0,
        'NoOfHyperlink': 0
    }
    
    html_content = ""
    loaded = False
    
    # Tenter de charger la page
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            html_bytes = response.read()
            # Tenter de décoder
            try:
                html_content = html_bytes.decode('utf-8', errors='ignore')
            except Exception:
                html_content = html_bytes.decode('latin1', errors='ignore')
            loaded = True
            logger.info(f"✓ Contenu HTML chargé avec succès pour {url} ({len(html_bytes)} octets)")
    except Exception as e:
        logger.warning(f"⚠ Impossible de charger la page HTML pour {url}: {e}. Utilisation des heuristiques URL uniquement.")
        
    if loaded and html_content:
        # Extraire les features depuis l'HTML
        lines = html_content.split('\n')
        html_features['LineLength'] = int(sum(len(l) for l in lines) / len(lines)) if lines else 0
        
        html_features['HasTitle'] = 1 if re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE) else 0
        html_features['HasMeta'] = 1 if re.search(r'<meta[^>]*>', html_content, re.IGNORECASE) else 0
        
        # Favicon
        html_features['HasFavicon'] = 1 if re.search(r'rel=["\'](shortcut )?icon["\']', html_content, re.IGNORECASE) else 0
        
        # External form submit
        forms = re.findall(r'<form[^>]*action=["\'](.*?)["\']', html_content, re.IGNORECASE)
        has_ext_submit = 0
        for form_action in forms:
            if form_action.startswith(("http://", "https://")):
                try:
                    action_domain = urlparse(form_action).netloc.lower()
                    if action_domain != domain:
                        has_ext_submit = 1
                        break
                except Exception:
                    pass
        html_features['HasExternalFormSubmit'] = has_ext_submit
        
        # Copyright
        html_features['HasCopyright'] = 1 if any(word in html_content.lower() for word in ["copyright", "©", "&copy;"]) else 0
        
        # Social Networks
        social_domains = ["facebook.com", "twitter.com", "linkedin.com", "instagram.com", "youtube.com"]
        html_features['HasSocialNetworking'] = 1 if any(sd in html_content.lower() for sd in social_domains) else 0
        
        # Password Field
        html_features['HasPasswordField'] = 1 if re.search(r'type=["\']password["\']', html_content, re.IGNORECASE) else 0
        
        # Submit Button
        html_features['HasSubmitButton'] = 1 if re.search(r'type=["\']submit["\']|<button[^>]*>', html_content, re.IGNORECASE) else 0
        
        # Keywords
        bank_keywords = ["bank", "credit", "banque", "finance", "login", "signin", "compte", "account"]
        pay_keywords = ["paypal", "payment", "paiement", "card", "carte", "visa", "mastercard", "transfer"]
        crypto_keywords = ["crypto", "bitcoin", "btc", "wallet", "binance", "metamask", "trustwallet"]
        
        html_features['HasKeywordBank'] = 1 if any(kw in html_content.lower() for kw in bank_keywords) else 0
        html_features['HasKeywordPay'] = 1 if any(kw in html_content.lower() for kw in pay_keywords) else 0
        html_features['HasKeywordCrypto'] = 1 if any(kw in html_content.lower() for kw in crypto_keywords) else 0
        
        # Counts
        html_features['NoOfPopup'] = len(re.findall(r'window\.open\(', html_content, re.IGNORECASE))
        html_features['NoOfiFrame'] = len(re.findall(r'<iframe[^>]*>', html_content, re.IGNORECASE))
        html_features['NoOfImage'] = len(re.findall(r'<img[^>]*>', html_content, re.IGNORECASE))
        html_features['NoOfJS'] = len(re.findall(r'<script[^>]*>', html_content, re.IGNORECASE))
        html_features['NoOfCSS'] = len(re.findall(r'<link[^>]*rel=["\']stylesheet["\']|<style[^>]*>', html_content, re.IGNORECASE))
        
        # Redirect
        html_features['NoOfURLRedirect'] = len(re.findall(r'http-equiv=["\']refresh["\']', html_content, re.IGNORECASE))
        html_features['NoOfHyperlink'] = len(re.findall(r'<a[^>]*href=', html_content, re.IGNORECASE))
        
    features.update(html_features)
    
    return features
