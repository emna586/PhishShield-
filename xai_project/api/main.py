"""
Point d'entrée principal de l'API FastAPI Phishing Detection + Dashboard XAI
"""

import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ajouter le chemin racine au path
sys.path.append(str(Path(__file__).parent.parent))

from api.routes import router as api_router
from utils import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Phishing Detection API",
    description="API hybride de détection de phishing avec explicabilité XAI (SHAP/LIME) et évaluation de robustesse (FGSM/PGD)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter le routeur API
app.include_router(api_router, prefix="/api", tags=["API Detection"])


# HTML Dashboard Intégré (Vanilla CSS, Glassmorphism, Outfit Font)
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhishShield — Dashboard Phishing XAI & Robustesse</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0b0f19;
            --bg-card: rgba(22, 28, 45, 0.7);
            --bg-input: rgba(30, 41, 59, 0.5);
            --border-color: rgba(255, 255, 255, 0.08);
            
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.35);
            --secondary: #a855f7;
            --secondary-glow: rgba(168, 85, 247, 0.3);
            
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.2);
            --danger: #ef4444;
            --danger-glow: rgba(239, 68, 68, 0.25);
            --warning: #f59e0b;
            --warning-glow: rgba(245, 158, 11, 0.2);
            
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dark: #64748b;
            
            --font-main: 'Outfit', sans-serif;
            --font-sub: 'Plus Jakarta Sans', sans-serif;
            
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-main);
            color: var(--text-main);
            font-family: var(--font-main);
            min-height: 100vh;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(at 10% 20%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(168, 85, 247, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
        }

        header {
            border-bottom: 1px solid var(--border-color);
            padding: 1.25rem 2rem;
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--text-main) 30%, var(--primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .logo span {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            padding: 0.2rem 0.6rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: white;
            -webkit-text-fill-color: white;
        }

        .nav-links {
            display: flex;
            gap: 1.5rem;
        }

        .nav-tab {
            color: var(--text-muted);
            text-decoration: none;
            font-weight: 500;
            font-size: 0.95rem;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: var(--transition);
        }

        .nav-tab:hover, .nav-tab.active {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.05);
        }

        .nav-tab.active {
            border-bottom: 2px solid var(--primary);
            border-radius: 8px 8px 0 0;
            background: rgba(99, 102, 241, 0.08);
        }

        main {
            flex: 1;
            padding: 2.5rem 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }

        .tab-content {
            display: none;
            animation: fadeIn 0.4s ease-out forwards;
        }

        .tab-content.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* --- CONTENEUR D'ANALYSE --- */
        .search-container {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(16px);
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }

        .search-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(to right, white, var(--text-muted));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .search-box {
            display: flex;
            gap: 1rem;
        }

        .search-input {
            flex: 1;
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            color: var(--text-main);
            font-family: var(--font-sub);
            font-size: 1rem;
            transition: var(--transition);
        }

        .search-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .btn {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 4px 15px var(--primary-glow);
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* --- SPINNER DE CHARGEMENT --- */
        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* --- SECTION DE RÉSULTAT --- */
        .result-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .result-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.75rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            margin-bottom: 2rem;
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            border-left: 4px solid var(--primary);
            padding-left: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* --- VOYANTS ET BADGES --- */
        .verdict-box {
            text-align: center;
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: var(--transition);
        }

        .verdict-box.legitimate {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
            border: 1px solid var(--success);
            box-shadow: 0 0 25px var(--success-glow);
        }

        .verdict-box.phishing {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
            border: 1px solid var(--danger);
            box-shadow: 0 0 25px var(--danger-glow);
        }

        .verdict-title {
            font-size: 1.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .verdict-box.legitimate .verdict-title { color: var(--success); }
        .verdict-box.phishing .verdict-title { color: var(--danger); }

        .verdict-desc {
            font-size: 0.95rem;
            color: var(--text-muted);
            line-height: 1.4;
        }

        .badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge.whitelist { background: var(--success-glow); color: var(--success); border: 1px solid var(--success); }
        .badge.blacklist { background: var(--danger-glow); color: var(--danger); border: 1px solid var(--danger); }
        .badge.heuristics { background: var(--warning-glow); color: var(--warning); border: 1px solid var(--warning); }
        .badge.hybrid { background: var(--primary-glow); color: var(--primary); border: 1px solid var(--primary); }

        /* --- STATS ET GAUGES --- */
        .gauge-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin: 1.5rem 0;
        }

        .circular-gauge {
            position: relative;
            width: 120px;
            height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: conic-gradient(var(--primary) 0deg, var(--bg-main) 0deg);
        }

        .circular-gauge::before {
            content: '';
            position: absolute;
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: #121824;
        }

        .gauge-value {
            position: relative;
            z-index: 10;
            font-size: 1.5rem;
            font-weight: 800;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .gauge-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 500;
            margin-top: 0.1rem;
        }

        .bar-metric {
            margin-bottom: 1.25rem;
        }

        .bar-label-container {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .bar-track {
            height: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            width: 0%;
            border-radius: 6px;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* --- GRAPHE XAI --- */
        .xai-flex {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .xai-flex {
                grid-template-columns: 1fr;
            }
        }

        .xai-chart {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }

        .xai-bar-row {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .xai-bar-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .xai-bar-bg {
            height: 24px;
            background: rgba(255,255,255,0.03);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
            display: flex;
            align-items: center;
        }

        .xai-bar-fill {
            height: 100%;
            width: 0%;
            transition: width 1s ease-out;
        }

        .xai-bar-fill.phishing {
            background: linear-gradient(to right, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.6));
            border-right: 3px solid var(--danger);
        }

        .xai-bar-fill.legitimate {
            background: linear-gradient(to right, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.6));
            border-right: 3px solid var(--success);
        }

        .xai-bar-label {
            position: absolute;
            left: 0.5rem;
            font-size: 0.75rem;
            font-family: var(--font-sub);
            font-weight: 600;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }
        
        .xai-bar-val {
            position: absolute;
            right: 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        /* --- HEURISTICS LIST --- */
        .rules-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }

        .rules-table th {
            text-align: left;
            padding: 0.75rem;
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
        }

        .rules-table td {
            padding: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            vertical-align: middle;
        }

        .rule-status {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            font-weight: bold;
            font-size: 0.75rem;
        }

        .rule-status.triggered {
            background: rgba(239, 68, 68, 0.15);
            color: var(--danger);
        }

        .rule-status.passed {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
        }

        /* --- ROBUSTNESS SEC --- */
        .rob-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }

        @media (max-width: 900px) {
            .rob-grid {
                grid-template-columns: 1fr;
            }
        }

        .rob-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            transition: var(--transition);
        }

        .rob-card:hover {
            border-color: var(--primary);
            transform: translateY(-4px);
        }

        .rob-model-name {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .rob-level {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
        }
        
        .rob-level.robuste { background: var(--success-glow); color: var(--success); }
        .rob-level.modere { background: var(--warning-glow); color: var(--warning); }
        .rob-level.vulnerable { background: var(--danger-glow); color: var(--danger); }

        .rob-stat-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 0.6rem;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid rgba(255,255,255,0.03);
        }

        .rob-stat-row:last-child {
            border: none;
            margin: 0;
            padding: 0;
        }

        .rob-stat-label {
            color: var(--text-muted);
        }

        .rob-stat-val {
            font-weight: 600;
        }

        .rob-stat-val.drop {
            color: var(--danger);
        }

        /* --- LIST MANAGEMENT --- */
        .list-manager {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 768px) {
            .list-manager {
                grid-template-columns: 1fr;
            }
        }

        .list-add-box {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .list-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            max-height: 300px;
            overflow-y: auto;
            padding: 0.5rem;
            border: 1px solid var(--border-color);
            background: rgba(0,0,0,0.1);
            border-radius: 12px;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.35rem 0.75rem;
            font-size: 0.85rem;
            font-family: var(--font-sub);
            transition: var(--transition);
        }

        .tag:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .tag-remove {
            cursor: pointer;
            color: var(--text-muted);
            font-weight: bold;
            font-size: 0.95rem;
            transition: var(--transition);
        }

        .tag-remove:hover {
            color: var(--danger);
        }
        
        .empty-list-msg {
            color: var(--text-dark);
            font-style: italic;
            font-size: 0.9rem;
            padding: 1rem;
            text-align: center;
            width: 100%;
        }

        /* Message d'erreur/succès pop-up */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            padding: 1rem 1.5rem;
            border-radius: 10px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            z-index: 1000;
            display: none;
            animation: slideIn 0.3s ease-out forwards;
            font-weight: 600;
            font-size: 0.95rem;
        }

        @keyframes slideIn {
            from { transform: translateY(100px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .toast.success { background: var(--success); color: white; }
        .toast.error { background: var(--danger); color: white; }
        
        /* Model Selector style */
        .model-select-wrapper {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        .model-select {
            background: var(--bg-input);
            color: white;
            border: 1px solid var(--border-color);
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
        }
        .model-select:focus {
            outline: none;
            border-color: var(--primary);
        }
    </style>
</head>
<body>

    <header>
        <div class="logo">
            PhishShield <span>XAI</span>
        </div>
        <div class="nav-links">
            <a class="nav-tab active" onclick="switchTab('tab-analyse', this)">Analyse URL</a>
            <a class="nav-tab" onclick="switchTab('tab-listes', this); loadLists();">Whitelist / Blacklist</a>
            <a class="nav-tab" onclick="switchTab('tab-docs', this)">Documentation</a>
        </div>
    </header>

    <main>
        <!-- TOAST MESSAGE -->
        <div id="toast" class="toast"></div>

        <!-- ONGLET 1: ANALYSE -->
        <div id="tab-analyse" class="tab-content active">
            <div class="search-container">
                <h2 class="search-title">Détection Hybride en Temps Réel</h2>
                <div class="search-box">
                    <input type="text" id="url-input" class="search-input" placeholder="Entrez l'URL à tester (ex: http://secure-login.bank-update.com)" value="">
                    <button id="btn-analyse" class="btn" onclick="analyseURL()">
                        <span>Analyser</span>
                        <div id="btn-spinner" class="spinner"></div>
                    </button>
                </div>
            </div>

            <!-- RÉSULTAT CACHÉ PAR DÉFAUT -->
            <div id="analysis-results-section" style="display: none;">
                <div class="result-grid">
                    <!-- COLONNE GAUCHE: VERDICT & GAUGES -->
                    <div>
                        <div class="card">
                            <div class="card-title">Verdict Global</div>
                            <div id="verdict-box" class="verdict-box legitimate">
                                <div id="verdict-title" class="verdict-title">Légitime</div>
                                <div id="verdict-badge" class="badge whitelist">Whitelist</div>
                                <div id="verdict-desc" class="verdict-desc">Le domaine fait partie de notre liste de confiance.</div>
                            </div>
                            
                            <div class="gauge-container">
                                <div class="circular-gauge" id="confidence-gauge">
                                    <div class="gauge-value">
                                        <span id="confidence-val">100%</span>
                                        <span class="gauge-label">Confiance</span>
                                    </div>
                                </div>
                            </div>

                            <div class="bar-metric">
                                <div class="bar-label-container">
                                    <span>Score Combiné (Dangerosité)</span>
                                    <span id="score-val">0.00%</span>
                                </div>
                                <div class="bar-track">
                                    <div id="score-bar" class="bar-fill"></div>
                                </div>
                            </div>
                        </div>

                        <!-- MODÈLES DE ML PROBABILITÉS -->
                        <div class="card">
                            <div class="card-title">Estimations Machine Learning</div>
                            <div id="ml-prob-container">
                                <!-- Injecté par JS -->
                            </div>
                        </div>
                    </div>

                    <!-- COLONNE DROITE: HEURISTIQUES / XAI / ROBUSTESSE -->
                    <div>
                        <!-- ONGLETS DE DÉTAILS -->
                        <div class="card">
                            <div class="card-title">
                                Explications XAI & Détails
                                <div class="model-select-wrapper" id="xai-model-wrapper" style="display: none;">
                                    Explication par :
                                    <select id="xai-model-select" class="model-select" onchange="analyseURL()">
                                        <option value="xgboost">XGBoost</option>
                                        <option value="lightgbm">LightGBM</option>
                                        <option value="random_forest">Random Forest</option>
                                    </select>
                                </div>
                            </div>

                            <!-- 1. Explications XAI en temps réel -->
                            <div id="xai-section" style="display: none; margin-bottom: 2rem;">
                                <h3 style="font-size: 1rem; margin-bottom: 1rem; color: var(--text-muted);">Attribution de Caractéristiques (SHAP vs LIME)</h3>
                                <div class="xai-flex">
                                    <!-- SHAP -->
                                    <div>
                                        <h4 style="font-size: 0.9rem; margin-bottom: 0.75rem; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.25rem;">SHAP (Global/Local)</h4>
                                        <div class="xai-chart" id="shap-chart">
                                            <!-- Injecté par JS -->
                                        </div>
                                    </div>
                                    <!-- LIME -->
                                    <div>
                                        <h4 style="font-size: 0.9rem; margin-bottom: 0.75rem; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.25rem;">LIME (Voisinage Local)</h4>
                                        <div class="xai-chart" id="lime-chart">
                                            <!-- Injecté par JS -->
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- 2. Règles heuristiques statiques -->
                            <div>
                                <h3 style="font-size: 1rem; margin-bottom: 0.75rem; color: var(--text-muted);">Évaluation des Règles Heuristiques (Max 85 points)</h3>
                                <table class="rules-table">
                                    <thead>
                                        <tr>
                                            <th>Règle</th>
                                            <th>Valeur</th>
                                            <th>Poids</th>
                                            <th>Statut</th>
                                        </tr>
                                    </thead>
                                    <tbody id="heuristics-table-body">
                                        <!-- Injecté par JS -->
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- SECTION ROBUSTESSE DES MODÈLES -->
                <div class="card">
                    <div class="card-title">🛡️ Résilience Adversariale (FGSM / PGD)</div>
                    <div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.5rem; line-height: 1.5;">
                        Les modèles de Machine Learning sont souvent vulnérables aux perturbations adversariales. Cette section montre les performances de nos modèles sous des attaques simulées <strong>FGSM</strong> (rapide, 1 étape) et <strong>PGD</strong> (itérative, 40 étapes, epsilon=0.1) sur le dataset d'évaluation.
                    </div>
                    <div class="rob-grid" id="robustness-grid">
                        <!-- Injecté par JS -->
                    </div>
                </div>
            </div>
        </div>

        <!-- ONGLET 2: LISTES -->
        <div id="tab-listes" class="tab-content">
            <div class="card">
                <div class="card-title">Gestion de la Whitelist & Blacklist</div>
                <div style="font-size: 0.95rem; color: var(--text-muted); margin-bottom: 2rem; line-height: 1.5;">
                    Les listes blanches (Whitelist) et noires (Blacklist) constituent la première couche défensive de notre système hybride. Tout domaine détecté dans ces listes reçoit immédiatement un verdict avec une confiance de 100%, court-circuitant les filtres heuristiques et le Machine Learning.
                </div>
                
                <div class="list-manager">
                    <!-- Whitelist -->
                    <div>
                        <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--success);">🟢 Whitelist (Confiance Légitime)</h3>
                        <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">Domaines exempts d'analyse, toujours classés légitimes.</p>
                        
                        <div class="list-add-box">
                            <input type="text" id="wl-add-input" class="search-input" style="padding: 0.6rem 1rem;" placeholder="Ajouter domaine (ex: netflix.com)">
                            <button class="btn" style="padding: 0.6rem 1.25rem;" onclick="addToList('whitelist')">Ajouter</button>
                        </div>
                        
                        <div class="list-tags" id="wl-tags-container">
                            <!-- Injecté par JS -->
                        </div>
                    </div>
                    
                    <!-- Blacklist -->
                    <div>
                        <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; color: var(--danger);">🔴 Blacklist (Confiance Phishing)</h3>
                        <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">Domaines de phishing connus, toujours bloqués.</p>
                        
                        <div class="list-add-box">
                            <input type="text" id="bl-add-input" class="search-input" style="padding: 0.6rem 1rem;" placeholder="Ajouter domaine (ex: secure-paypal.net)">
                            <button class="btn" style="padding: 0.6rem 1.25rem; background: linear-gradient(135deg, var(--danger) 0%, #b91c1c 100%); box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);" onclick="addToList('blacklist')">Ajouter</button>
                        </div>
                        
                        <div class="list-tags" id="bl-tags-container">
                            <!-- Injecté par JS -->
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ONGLET 3: DOCUMENTATION -->
        <div id="tab-docs" class="tab-content">
            <div class="card">
                <div class="card-title">Informations Techniques et Méthodologie</div>
                <div style="font-size: 0.95rem; line-height: 1.6; color: var(--text-muted);">
                    <h3 style="color: white; margin-bottom: 0.5rem; font-size: 1.15rem;">Architecture du Système Hybride</h3>
                    <p style="margin-bottom: 1.25rem;">
                        Le système combine 3 approches complémentaires pour garantir un excellent taux de détection tout en conservant une très faible latence :
                    </p>
                    <ul style="margin-left: 1.5rem; margin-bottom: 1.5rem; display: flex; flex-direction: column; gap: 0.5rem;">
                        <li><strong>Couche 1 : Whitelist / Blacklist</strong> — Comparaison de chaîne ultra-rapide sur le domaine racine. Décision instantanée en cas de correspondance.</li>
                        <li><strong>Couche 2 : Règles Heuristiques</strong> — Analyse structurelle statique basée sur l'URL brute (compte de points, sous-domaines, chiffres, protocole, etc.). Total des règles sur 85 points.</li>
                        <li><strong>Couche 3 : Machine Learning Ensembliste</strong> — Si non décidé par liste, les caractéristiques de l'URL et de son code HTML sont extraites et envoyées à 3 classifieurs entraînés (XGBoost, LightGBM, Random Forest).</li>
                    </ul>

                    <h3 style="color: white; margin-bottom: 0.5rem; font-size: 1.15rem;">Explicabilité (XAI)</h3>
                    <p style="margin-bottom: 1.25rem;">
                        Pour justifier la décision du modèle de Machine Learning, nous intégrons 2 méthodes XAI majeures :
                    </p>
                    <ul style="margin-left: 1.5rem; margin-bottom: 1.5rem; display: flex; flex-direction: column; gap: 0.5rem;">
                        <li><strong>SHAP (SHapley Additive exPlanations)</strong> : Approche basée sur la théorie des jeux pour attribuer équitablement l'impact de chaque caractéristique sur la prédiction finale du modèle.</li>
                        <li><strong>LIME (Local Interpretable Model-agnostic Explanations)</strong> : Approche par modèle de substitution local (surrogate), approximant les contours de décision locaux du modèle complexe pour expliquer un échantillon individuel.</li>
                    </ul>

                    <h3 style="color: white; margin-bottom: 0.5rem; font-size: 1.15rem;">Robustesse face aux attaques</h3>
                    <p>
                        Les attaquants cherchent souvent à modifier légèrement les URLs de phishing (par exemple en allongeant l'URL ou en ajoutant des paramètres) pour tromper les modèles. Notre module de robustesse évalue la résilience de nos modèles contre les attaques de type perturbation de signe par gradient (<strong>FGSM</strong>) et descente de gradient projetée (<strong>PGD</strong>).
                    </p>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">Points d'accès de l'API (Endpoints)</div>
                <div style="font-size: 0.9rem; font-family: monospace; background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 1rem;">
                    <div>
                        <span style="color: #60a5fa; font-weight: bold;">POST</span> /api/predict<br>
                        <span style="color: var(--text-muted);">Request body:</span> { "url": "string" }<br>
                        <span style="color: var(--text-muted);">Description:</span> Analyse l'URL et renvoie le verdict avec XAI et robustesse.
                    </div>
                    <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.05);">
                    <div>
                        <span style="color: #34d399; font-weight: bold;">GET</span> /api/lists<br>
                        <span style="color: var(--text-muted);">Description:</span> Récupère la Whitelist et la Blacklist.
                    </div>
                    <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.05);">
                    <div>
                        <span style="color: #60a5fa; font-weight: bold;">POST</span> /api/lists/add<br>
                        <span style="color: var(--text-muted);">Request body:</span> { "domain": "string", "list_type": "string" }<br>
                        <span style="color: var(--text-muted);">Description:</span> Ajoute un domaine à la whitelist ou la blacklist.
                    </div>
                    <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.05);">
                    <div>
                        <span style="color: #60a5fa; font-weight: bold;">POST</span> /api/lists/remove<br>
                        <span style="color: var(--text-muted);">Request body:</span> { "domain": "string", "list_type": "string" }<br>
                        <span style="color: var(--text-muted);">Description:</span> Retire un domaine d'une liste.
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        // Fonction pour changer d'onglet
        function switchTab(tabId, el) {
            // Désactiver tous les onglets
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(nt => nt.classList.remove('active'));
            
            // Activer l'onglet sélectionné
            document.getElementById(tabId).classList.add('active');
            el.classList.add('active');
        }

        // Système d'alerte Toast
        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = `toast ${type}`;
            toast.style.display = 'block';
            
            setTimeout(() => {
                toast.style.display = 'none';
            }, 3000);
        }

        // --- ENVOYER L'URL POUR ANALYSE ---
        async function analyseURL() {
            const urlInput = document.getElementById('url-input').value.trim();
            const btn = document.getElementById('btn-analyse');
            const spinner = document.getElementById('btn-spinner');
            const modelSelect = document.getElementById('xai-model-select').value;
            
            if (!urlInput) {
                showToast("Veuillez entrer une URL valide", "error");
                return;
            }

            // Afficher le loading
            btn.disabled = true;
            spinner.style.display = 'block';

            try {
                const response = await fetch(`/api/predict?model=${modelSelect}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: urlInput })
                });

                if (!response.ok) {
                    throw new Error("Erreur de communication avec le serveur.");
                }

                const data = await response.json();
                
                // Mettre à jour l'interface
                displayResults(data);
                showToast("Analyse terminée !");
            } catch (err) {
                showToast(err.message, "error");
            } finally {
                // Arrêter le loading
                btn.disabled = false;
                spinner.style.display = 'none';
            }
        }

        // --- AFFICHER LES RÉSULTATS DYNAMIQUE ---
        function displayResults(data) {
            document.getElementById('analysis-results-section').style.display = 'block';
            
            // 1. Verdict Box
            const verdictBox = document.getElementById('verdict-box');
            const verdictTitle = document.getElementById('verdict-title');
            const verdictBadge = document.getElementById('verdict-badge');
            const verdictDesc = document.getElementById('verdict-desc');
            
            verdictBox.className = `verdict-box ${data.status}`;
            if (data.status === 'phishing') {
                verdictTitle.textContent = 'Phishing / Risque élevé';
                verdictBadge.textContent = data.method;
                verdictBadge.className = `badge ${data.method === 'blacklist' ? 'blacklist' : (data.method === 'heuristics' ? 'heuristics' : 'hybrid')}`;
            } else {
                verdictTitle.textContent = 'Légitime / Sûr';
                verdictBadge.textContent = data.method;
                verdictBadge.className = `badge ${data.method === 'whitelist' ? 'whitelist' : 'hybrid'}`;
            }
            verdictDesc.textContent = data.verdict_details;
            
            // 2. Jauges et Scores
            const confPercent = Math.round(data.confidence * 100);
            document.getElementById('confidence-val').textContent = `${confPercent}%`;
            
            // Calcul du conic-gradient
            const color = data.status === 'phishing' ? '#ef4444' : '#10b981';
            document.getElementById('confidence-gauge').style.background = `conic-gradient(${color} ${confPercent * 3.6}deg, rgba(255,255,255,0.05) 0deg)`;
            
            document.getElementById('score-val').textContent = `${data.score.toFixed(2)}%`;
            document.getElementById('score-bar').style.width = `${data.score}%`;
            // Ajuster la couleur du score bar
            document.getElementById('score-bar').style.background = data.status === 'phishing' ? 'linear-gradient(to right, #f59e0b, #ef4444)' : 'linear-gradient(to right, #10b981, #34d399)';

            // 3. Estimations ML
            const mlContainer = document.getElementById('ml-prob-container');
            mlContainer.innerHTML = '';
            
            if (Object.keys(data.ml_probabilities).length > 0) {
                for (const [model, prob] of Object.entries(data.ml_probabilities)) {
                    const probPercent = Math.round(prob * 100);
                    const modelName = model === 'xgboost' ? 'XGBoost' : (model === 'lightgbm' ? 'LightGBM' : 'Random Forest');
                    
                    const itemHtml = `
                        <div class="bar-metric" style="margin-bottom: 0.75rem;">
                            <div class="bar-label-container" style="font-size: 0.8rem;">
                                <span>${modelName}</span>
                                <span>${probPercent}% Phishing</span>
                            </div>
                            <div class="bar-track" style="height: 6px;">
                                <div class="bar-fill" style="width: ${probPercent}%; background: ${probPercent >= 50 ? 'var(--danger)' : 'var(--success)'};"></div>
                            </div>
                        </div>
                    `;
                    mlContainer.insertAdjacentHTML('beforeend', itemHtml);
                }
            } else {
                mlContainer.innerHTML = '<div class="empty-list-msg">Aucune prédiction ML (décidé par whitelist/blacklist).</div>';
            }

            // 4. Section XAI (SHAP & LIME)
            const xaiSection = document.getElementById('xai-section');
            const xaiModelWrapper = document.getElementById('xai-model-wrapper');
            
            if (data.explanations && data.explanations.shap.length > 0) {
                xaiSection.style.display = 'block';
                xaiModelWrapper.style.display = 'block';
                
                // Dessiner SHAP
                const shapChart = document.getElementById('shap-chart');
                shapChart.innerHTML = '';
                
                // Trouver la valeur d'importance max pour normaliser les tailles des barres
                const maxShap = Math.max(...data.explanations.shap.map(x => x.importance), 0.001);
                
                data.explanations.shap.forEach(item => {
                    const width = (item.importance / maxShap) * 100;
                    const c = item.effect === 'Favorise Phishing' ? 'phishing' : 'legitimate';
                    const sign = item.effect === 'Favorise Phishing' ? '+' : '-';
                    const rowHtml = `
                        <div class="xai-bar-row">
                            <div class="xai-bar-header">
                                <span>${item.feature} (${item.value})</span>
                                <span>${sign}${item.importance.toFixed(3)}</span>
                            </div>
                            <div class="xai-bar-bg">
                                <div class="xai-bar-fill ${c}" style="width: ${width}%;"></div>
                                <span class="xai-bar-label">${item.effect}</span>
                            </div>
                        </div>
                    `;
                    shapChart.insertAdjacentHTML('beforeend', rowHtml);
                });

                // Dessiner LIME
                const limeChart = document.getElementById('lime-chart');
                limeChart.innerHTML = '';
                
                const maxLime = Math.max(...data.explanations.lime.map(x => x.importance), 0.001);
                
                data.explanations.lime.forEach(item => {
                    const width = (item.importance / maxLime) * 100;
                    const c = item.effect === 'Favorise Phishing' ? 'phishing' : 'legitimate';
                    const sign = item.effect === 'Favorise Phishing' ? '+' : '-';
                    const rowHtml = `
                        <div class="xai-bar-row">
                            <div class="xai-bar-header">
                                <span>${item.feature} (${item.value})</span>
                                <span>${sign}${item.importance.toFixed(3)}</span>
                            </div>
                            <div class="xai-bar-bg">
                                <div class="xai-bar-fill ${c}" style="width: ${width}%;"></div>
                                <span class="xai-bar-label">${item.effect}</span>
                            </div>
                        </div>
                    `;
                    limeChart.insertAdjacentHTML('beforeend', rowHtml);
                });
            } else {
                xaiSection.style.display = 'none';
                xaiModelWrapper.style.display = 'none';
            }

            // 5. Heuristics Table
            const heurBody = document.getElementById('heuristics-table-body');
            heurBody.innerHTML = '';
            
            for (const [rule, info] of Object.entries(data.heuristics.details)) {
                const badge = info.triggered ? '<span class="rule-status triggered">✕</span>' : '<span class="rule-status passed">✓</span>';
                const rowHtml = `
                    <tr>
                        <td style="font-weight: 500;">${rule}</td>
                        <td>${info.value}</td>
                        <td style="font-weight: bold; color: ${info.triggered ? 'var(--danger)' : 'var(--text-dark)'};">${info.points} pts</td>
                        <td>${badge} <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 0.5rem;">${info.description}</span></td>
                    </tr>
                `;
                heurBody.insertAdjacentHTML('beforeend', rowHtml);
            }
            // Ligne totale score
            const totalHtml = `
                <tr style="background: rgba(255,255,255,0.02); font-weight: bold; border-top: 2px solid var(--border-color);">
                    <td>Total Heuristique</td>
                    <td>—</td>
                    <td style="color: var(--warning); font-size: 1rem;">${data.heuristics.score} / ${data.heuristics.max_score} pts</td>
                    <td style="color: var(--text-muted);">${data.heuristics.score_percent.toFixed(1)}% de dangerosité statique</td>
                </tr>
            `;
            heurBody.insertAdjacentHTML('beforeend', totalHtml);

            // 6. Robustness Info
            const robGrid = document.getElementById('robustness-grid');
            robGrid.innerHTML = '';
            
            for (const [model, info] of Object.entries(data.robustness)) {
                const modelName = model === 'xgboost' ? 'XGBoost' : (model === 'lightgbm' ? 'LightGBM' : 'Random Forest');
                const levelClass = info.pgd_drop <= 0.05 ? 'robuste' : (info.pgd_drop <= 0.15 ? 'modere' : 'vulnerable');
                const levelText = info.pgd_drop <= 0.05 ? '🟢 ROBUSTE' : (info.pgd_drop <= 0.15 ? '🟡 MODÉRÉ' : '🟠 VULNÉRABLE');
                
                const cardHtml = `
                    <div class="rob-card">
                        <div class="rob-model-name">
                            <span>${modelName}</span>
                            <span class="rob-level ${levelClass}">${levelText}</span>
                        </div>
                        
                        <div class="rob-stat-row">
                            <span class="rob-stat-label">Chute FGSM (accuracy drop)</span>
                            <span class="rob-stat-val drop">${(info.fgsm_drop * 100).toFixed(1)}%</span>
                        </div>
                        <div class="rob-stat-row">
                            <span class="rob-stat-label">Chute PGD (40 itér.)</span>
                            <span class="rob-stat-val drop">${(info.pgd_drop * 100).toFixed(1)}%</span>
                        </div>
                        <div class="rob-stat-row">
                            <span class="rob-stat-label">Score de Robustesse</span>
                            <span class="rob-stat-val" style="color: var(--success); font-weight: bold;">${(info.robustness_score * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                `;
                robGrid.insertAdjacentHTML('beforeend', cardHtml);
            }
        }

        // --- GESTION DES LISTES (AJOUT / RETRAIT) ---
        async function loadLists() {
            try {
                const response = await fetch('/api/lists');
                const data = await response.json();
                
                // Afficher whitelist
                const wlContainer = document.getElementById('wl-tags-container');
                wlContainer.innerHTML = '';
                if (data.whitelist.length > 0) {
                    data.whitelist.forEach(domain => {
                        wlContainer.insertAdjacentHTML('beforeend', `
                            <span class="tag">
                                ${domain}
                                <span class="tag-remove" onclick="removeFromList('${domain}', 'whitelist')">&times;</span>
                            </span>
                        `);
                    });
                } else {
                    wlContainer.innerHTML = '<div class="empty-list-msg">Aucun domaine dans la whitelist.</div>';
                }

                // Afficher blacklist
                const blContainer = document.getElementById('bl-tags-container');
                blContainer.innerHTML = '';
                if (data.blacklist.length > 0) {
                    data.blacklist.forEach(domain => {
                        blContainer.insertAdjacentHTML('beforeend', `
                            <span class="tag">
                                ${domain}
                                <span class="tag-remove" style="color: rgba(239, 68, 68, 0.6);" onclick="removeFromList('${domain}', 'blacklist')">&times;</span>
                            </span>
                        `);
                    });
                } else {
                    blContainer.innerHTML = '<div class="empty-list-msg">Aucun domaine dans la blacklist.</div>';
                }
            } catch (err) {
                showToast("Erreur lors du chargement des listes", "error");
            }
        }

        async function addToList(listType) {
            const inputId = listType === 'whitelist' ? 'wl-add-input' : 'bl-add-input';
            const domain = document.getElementById(inputId).value.trim();
            
            if (!domain) {
                showToast("Veuillez entrer un domaine valide", "error");
                return;
            }

            try {
                const response = await fetch('/api/lists/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ domain, list_type: listType })
                });
                
                const resData = await response.json();
                if (resData.success) {
                    showToast(resData.message);
                    document.getElementById(inputId).value = '';
                    loadLists();
                } else {
                    showToast(resData.message, "error");
                }
            } catch (err) {
                showToast("Erreur lors de l'ajout", "error");
            }
        }

        async function removeFromList(domain, listType) {
            try {
                const response = await fetch('/api/lists/remove', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ domain, list_type: listType })
                });
                
                const resData = await response.json();
                if (resData.success) {
                    showToast(resData.message);
                    loadLists();
                } else {
                    showToast(resData.message, "error");
                }
            } catch (err) {
                showToast("Erreur lors de la suppression", "error");
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, summary="Dashboard de détection de phishing")
def get_dashboard():
    """
    Sert le tableau de bord HTML interactif de détection de phishing.
    """
    return HTMLResponse(content=DASHBOARD_HTML)


if __name__ == "__main__":
    logger.info("Démarrage du serveur FastAPI PhishShield...")
    uvicorn.run(app, host="localhost", port=8000)
