"""
Module Whitelist/Blacklist pour la détection de phishing.
Gère les listes de domaines de confiance (whitelist) et de phishing connus (blacklist).
"""

import json
from pathlib import Path
from urllib.parse import urlparse
from utils import setup_logger

logger = setup_logger(__name__)


class WhitelistBlacklist:
    """
    Gère la Whitelist (sites légitimes connus) et la Blacklist (sites de phishing connus)
    avec sauvegarde persistante au format JSON.

    La whitelist est initialisée avec les grands domaines légitimes.
    La blacklist est initialisée avec des domaines de phishing connus.
    """

    # Whitelist étendue avec les domaines les plus fiables
    DEFAULT_WHITELIST = {
        # Moteurs de recherche
        "google.com", "google.fr", "bing.com", "yahoo.com", "duckduckgo.com",
        # Réseaux sociaux
        "facebook.com", "instagram.com", "twitter.com", "x.com",
        "linkedin.com", "youtube.com", "tiktok.com", "snapchat.com",
        # Tech
        "microsoft.com", "apple.com", "amazon.com", "amazon.fr",
        "github.com", "stackoverflow.com", "wikipedia.org",
        # Email
        "gmail.com", "outlook.com", "hotmail.com", "yahoo.fr",
        # Banques FR
        "bnpparibas.fr", "societegenerale.fr", "creditagricole.fr",
        "labanquepostale.fr", "lcl.fr", "caisse-epargne.fr",
        # Services publics FR
        "service-public.fr", "impots.gouv.fr", "ameli.fr",
        "caf.fr", "pole-emploi.fr", "elysee.fr",
        # Paiement
        "paypal.com", "stripe.com", "visa.com", "mastercard.com",
    }

    # Blacklist avec domaines de phishing réels connus
    DEFAULT_BLACKLIST = {
        "phishing-bank.com",
        "fake-google.com",
        "secure-bank-login.com",
        "paypal-verification.com",
        "login-paypal-secure.com",
        "update-your-account.com",
        "account-suspended-verify.com",
        "secure-login-microsoft.net",
        "amazon-security-alert.com",
    }

    def __init__(self, filepath: str = "outputs/whitelist_blacklist.json"):
        """
        Initialise les listes avec des valeurs par défaut et charge le fichier JSON s'il existe.

        Args:
            filepath: Chemin du fichier JSON de persistence.
        """
        self.filepath  = Path(filepath)
        self.whitelist = set(self.DEFAULT_WHITELIST)
        self.blacklist = set(self.DEFAULT_BLACKLIST)
        self.load()

    def load(self) -> None:
        """
        Charge la whitelist et la blacklist depuis le fichier JSON.
        Si le fichier n'existe pas, sauvegarde les valeurs par défaut.
        """
        try:
            if self.filepath.exists():
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    loaded_wl = set(data.get("whitelist", []))
                    loaded_bl = set(data.get("blacklist", []))
                    # Fusionner avec les defaults pour ne rien perdre
                    self.whitelist = self.DEFAULT_WHITELIST | loaded_wl
                    self.blacklist = self.DEFAULT_BLACKLIST | loaded_bl
                logger.info(f"[OK] Whitelist ({len(self.whitelist)}) / Blacklist ({len(self.blacklist)}) chargees.")
            else:
                # Première fois : sauvegarder les defaults
                self.save()
        except Exception as e:
            logger.error(f"[ERR] Erreur lors du chargement des listes: {e}", exc_info=True)

    def save(self) -> None:
        """
        Sauvegarde la whitelist et la blacklist dans le fichier JSON.
        """
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "whitelist": sorted(list(self.whitelist)),
                    "blacklist": sorted(list(self.blacklist))
                }, f, indent=4, ensure_ascii=False)
            logger.info(f"[OK] Whitelist/Blacklist sauvegardees dans {self.filepath}")
        except Exception as e:
            logger.error(f"[ERR] Erreur lors de la sauvegarde des listes: {e}", exc_info=True)

    def get_domain(self, url: str) -> str:
        """
        Extrait le nom de domaine principal (sans www. ni port) depuis une URL.

        Args:
            url: L'URL à analyser.

        Returns:
            Le nom de domaine en minuscules, sans 'www.' ni port.
        """
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Retirer le port
            if ":" in domain:
                domain = domain.split(":")[0]
            # Retirer le préfixe www.
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception as e:
            logger.warning(f"Impossible d'extraire le domaine de '{url}': {e}")
            return ""

    def check_whitelist(self, url: str) -> bool:
        """
        Vérifie si le domaine ou un parent du domaine est dans la whitelist.
        Ex: 'mail.google.com' → vérifie aussi 'google.com'
        """
        domain = self.get_domain(url)
        if not domain:
            return False
        # Vérification exacte
        if domain in self.whitelist:
            return True
        # Vérification du domaine parent (ex: 'accounts.google.com' → 'google.com')
        parts = domain.split(".")
        if len(parts) >= 2:
            parent = ".".join(parts[-2:])
            if parent in self.whitelist:
                return True
        return False

    def check_blacklist(self, url: str) -> bool:
        """
        Vérifie si le domaine est dans la blacklist.
        """
        domain = self.get_domain(url)
        return domain in self.blacklist

    def add_to_whitelist(self, domain: str) -> bool:
        """
        Ajoute un domaine à la whitelist et le retire de la blacklist si présent.

        Returns:
            True si l'ajout a réussi, False sinon.
        """
        domain = domain.lower().strip()
        if not domain:
            return False
        self.whitelist.add(domain)
        self.blacklist.discard(domain)
        self.save()
        logger.info(f"[OK] '{domain}' ajoute a la whitelist.")
        return True

    def add_to_blacklist(self, domain: str) -> bool:
        """
        Ajoute un domaine à la blacklist et le retire de la whitelist si présent.

        Returns:
            True si l'ajout a réussi, False sinon.
        """
        domain = domain.lower().strip()
        if not domain:
            return False
        self.blacklist.add(domain)
        self.whitelist.discard(domain)
        self.save()
        logger.info(f"[OK] '{domain}' ajoute a la blacklist.")
        return True

    def remove_from_whitelist(self, domain: str) -> bool:
        """
        Supprime un domaine de la whitelist.
        """
        domain = domain.lower().strip()
        if domain in self.whitelist:
            self.whitelist.remove(domain)
            self.save()
            return True
        return False

    def remove_from_blacklist(self, domain: str) -> bool:
        """
        Supprime un domaine de la blacklist.
        """
        domain = domain.lower().strip()
        if domain in self.blacklist:
            self.blacklist.remove(domain)
            self.save()
            return True
        return False

    def get_stats(self) -> dict:
        """
        Retourne les statistiques des listes.
        """
        return {
            "whitelist_count": len(self.whitelist),
            "blacklist_count": len(self.blacklist),
            "whitelist_domains": sorted(list(self.whitelist)),
            "blacklist_domains": sorted(list(self.blacklist)),
        }
