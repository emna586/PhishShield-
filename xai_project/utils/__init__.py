"""
Utilitaires pour le projet XAI Phishing Detection.
Configure le logger avec encodage UTF-8 pour la console et le fichier.
"""

import logging
import sys
from pathlib import Path
from config import LOGGING_CONFIG, LOGS_DIR


def setup_logger(name: str) -> logging.Logger:
    """
    Configure et retourne un logger avec encodage UTF-8.
    Sur Windows (console CP1252), le handler console utilise 'replace'
    pour éviter les UnicodeEncodeError avec les caractères spéciaux.

    Args:
        name: Nom du logger (généralement __name__)

    Returns:
        logging.Logger configuré
    """
    logger = logging.getLogger(name)

    # Éviter de reconfigurer plusieurs fois le même logger
    if logger.handlers:
        return logger

    logger.setLevel(LOGGING_CONFIG['level'])

    # ── Handler Fichier (toujours UTF-8) ─────────────────────────────────────
    try:
        file_handler = logging.FileHandler(
            LOGGING_CONFIG['log_file'],
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
    except Exception:
        file_handler = None

    # ── Handler Console (UTF-8 avec replace sur Windows) ─────────────────────
    try:
        if sys.platform == "win32":
            import io
            # Envelopper stdout avec encodage UTF-8 et remplacement des caractères inconnus
            utf8_stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
            console_handler = logging.StreamHandler(utf8_stdout)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
    except Exception:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

    # ── Format commun ─────────────────────────────────────────────────────────
    formatter = logging.Formatter(LOGGING_CONFIG['format'])

    if file_handler:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def ensure_dir_exists(path: Path) -> None:
    """
    Crée un répertoire s'il n'existe pas.

    Args:
        path: Chemin du répertoire à créer
    """
    path.mkdir(parents=True, exist_ok=True)


# Logger global pour ce module
logger = setup_logger(__name__)
