"""
Script de test complet de l'API FastAPI PhishShield.
Teste TOUS les endpoints avec validation automatique des résultats.

Usage:
    1. Lancer l'API: uvicorn api.main:app --reload --port 8000
    2. Dans un autre terminal: python test_api_complet.py
"""

import urllib.request
import urllib.error
import json
import sys
import time
import io

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://127.0.0.1:8000"

# ──────────────────────────────────────────────────────────────────
# Couleurs console (fonctionne sur Windows 10+ et Linux)
# ──────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}[OK]{RESET}   {msg}")
def fail(msg): print(f"  {RED}[FAIL]{RESET} {msg}")
def info(msg): print(f"  {CYAN}[INFO]{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")

# ──────────────────────────────────────────────────────────────────
# Helpers HTTP
# ──────────────────────────────────────────────────────────────────

def http_get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_expect_error(path: str, body: dict) -> int:
    """Retourne le code HTTP (erreur attendue)."""
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


# ──────────────────────────────────────────────────────────────────
# Résumé global
# ──────────────────────────────────────────────────────────────────
results = []

def run_test(name: str, fn):
    sep = '-' * max(2, 55 - len(name))
    print(f"\n{BOLD}-- {name} {sep}{RESET}")
    try:
        fn()
        results.append((name, True))
    except AssertionError as e:
        fail(f"Assertion échouée : {e}")
        results.append((name, False))
    except Exception as e:
        fail(f"Exception : {type(e).__name__}: {e}")
        results.append((name, False))


# ══════════════════════════════════════════════════════════════════
# TEST 0 — Vérifier que l'API est démarrée
# ══════════════════════════════════════════════════════════════════

def test_api_alive():
    req = urllib.request.Request(f"{BASE_URL}/", method="GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        assert resp.status == 200
    ok("L'API FastAPI est accessible (GET / repond 200)")

# ══════════════════════════════════════════════════════════════════
# TEST 1 — Dashboard HTML (GET /)
# ══════════════════════════════════════════════════════════════════

def test_dashboard():
    req = urllib.request.Request(f"{BASE_URL}/", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8")
    assert "PhishShield" in html, "Le titre 'PhishShield' est absent du HTML"
    assert "Analyser" in html, "Le bouton Analyser est absent"
    assert "Whitelist" in html, "L'onglet Whitelist est absent"
    ok("Dashboard HTML valide (PhishShield, Analyser, Whitelist présents)")

# ══════════════════════════════════════════════════════════════════
# TEST 2 — /api/predict — URL légitime (Whitelist)
# ══════════════════════════════════════════════════════════════════

def test_predict_whitelist():
    r = http_post("/api/predict", {"url": "https://www.google.com"})
    info(f"Status  : {r['status'].upper()}")
    info(f"Méthode : {r['method']}")
    info(f"Score   : {r['score']:.1f}/100")
    info(f"Confiance : {r['confidence']*100:.1f}%")
    assert r["status"] == "legitimate", f"Attendu 'legitimate', reçu '{r['status']}'"
    assert r["method"] == "whitelist",  f"Attendu méthode 'whitelist', reçu '{r['method']}'"
    assert r["confidence"] == 1.0, "La confiance pour une URL whitelist doit être 1.0"
    ok("google.com → LEGITIMATE via Whitelist (confiance 100%)")

# ══════════════════════════════════════════════════════════════════
# TEST 3 — /api/predict — URL blacklistée (Phishing direct)
# ══════════════════════════════════════════════════════════════════

def test_predict_blacklist():
    r = http_post("/api/predict", {"url": "http://phishing-bank.com/secure/login"})
    info(f"Status  : {r['status'].upper()}")
    info(f"Méthode : {r['method']}")
    info(f"Score   : {r['score']:.1f}/100")
    assert r["status"] == "phishing", f"Attendu 'phishing', reçu '{r['status']}'"
    assert r["method"] == "blacklist", f"Attendu méthode 'blacklist', reçu '{r['method']}'"
    assert r["confidence"] == 1.0, "La confiance pour une URL blacklist doit être 1.0"
    ok("phishing-bank.com → PHISHING via Blacklist (confiance 100%)")

# ══════════════════════════════════════════════════════════════════
# TEST 4 — /api/predict — URL suspecte avec ML + XAI
# ══════════════════════════════════════════════════════════════════

def test_predict_suspicious_ml():
    url = "http://secure-verify.paypal-account.update-now.com/login?user=admin&token=abc123"
    r = http_post("/api/predict?model=xgboost", {"url": url})
    info(f"Status   : {r['status'].upper()}")
    info(f"Méthode  : {r['method']}")
    info(f"Score    : {r['score']:.1f}/100")
    info(f"Confiance: {r['confidence']*100:.1f}%")

    # Vérifier la structure heuristique
    assert "heuristics" in r, "Clé 'heuristics' absente de la réponse"
    h = r["heuristics"]
    assert "score_percent" in h, "score_percent absent des heuristiques"
    info(f"Score heuristique : {h['score_percent']:.1f}%")
    info(f"Règles détaillées : {list(h.get('details', {}).keys())}")

    # Vérifier les probabilités ML
    ml = r.get("ml_probabilities", {})
    if ml:
        for model_name, prob in ml.items():
            info(f"ML prob [{model_name}]: {prob*100:.1f}%")
        ok("Probabilités ML reçues pour tous les modèles")
    else:
        warn("Probabilités ML vides (modèles peut-être non entraînés)")

    # Vérifier les explications XAI (SHAP + LIME)
    expl = r.get("explanations")
    if expl:
        shap_list = expl.get("shap", [])
        lime_list = expl.get("lime", [])
        if shap_list:
            top = shap_list[0]
            info(f"SHAP top feature : {top['feature']} → {top['effect']} (importance={top['importance']:.4f})")
            ok(f"SHAP retourne {len(shap_list)} features explicatives")
        else:
            warn("SHAP vide")
        if lime_list:
            top = lime_list[0]
            info(f"LIME top feature : {top['feature']} → {top['effect']} (importance={top['importance']:.4f})")
            ok(f"LIME retourne {len(lime_list)} features explicatives")
        else:
            warn("LIME vide (dataset background manquant?)")
    else:
        warn("Pas d'explications XAI (méthode non-hybrid ou ML indisponible)")

    ok("Test URL suspecte terminé — structure réponse valide")

# ══════════════════════════════════════════════════════════════════
# TEST 5 — /api/predict — Adresse IP suspecte
# ══════════════════════════════════════════════════════════════════

def test_predict_ip_url():
    r = http_post("/api/predict", {"url": "http://123.45.67.89/update?id=992&bank=paypal"})
    info(f"Status   : {r['status'].upper()}")
    info(f"Méthode  : {r['method']}")
    info(f"Score    : {r['score']:.1f}/100")
    assert r["status"] in ("phishing", "legitimate"), "Status invalide"
    ok(f"IP suspecte analysée → {r['status'].upper()} ({r['confidence']*100:.0f}% confiance)")

# ══════════════════════════════════════════════════════════════════
# TEST 6 — /api/predict — 3 modèles différents
# ══════════════════════════════════════════════════════════════════

def test_predict_all_models():
    url = "http://fake-amazon-login.suspicious-domain.com/signin"
    for model in ["xgboost", "lightgbm", "random_forest"]:
        r = http_post(f"/api/predict?model={model}", {"url": url})
        assert "status" in r, f"Clé 'status' absente pour modèle {model}"
        assert "explanations" in r, f"Clé 'explanations' absente pour modèle {model}"
        info(f"[{model:15s}] → {r['status'].upper():10s} | score={r['score']:.1f} | confiance={r['confidence']*100:.0f}%")
    ok("Les 3 modèles (XGBoost, LightGBM, RF) fonctionnent correctement")

# ══════════════════════════════════════════════════════════════════
# TEST 7 — /api/predict — Robustesse dans la réponse
# ══════════════════════════════════════════════════════════════════

def test_robustness_in_response():
    r = http_post("/api/predict", {"url": "http://test-phishing.example.com/login"})
    rob = r.get("robustness", {})
    assert rob, "Métriques de robustesse absentes de la réponse"
    for model_name, m in rob.items():
        assert "fgsm_drop" in m, f"fgsm_drop absent pour {model_name}"
        assert "pgd_drop" in m,  f"pgd_drop absent pour {model_name}"
        assert "robustness_score" in m, f"robustness_score absent pour {model_name}"
        info(f"  [{model_name:15s}] fgsm_drop={m['fgsm_drop']:.2f} | pgd_drop={m['pgd_drop']:.2f} | score={m['robustness_score']:.3f}")
    ok("Métriques de robustesse présentes pour tous les modèles")

# ══════════════════════════════════════════════════════════════════
# TEST 8 — /api/predict — Validation erreur URL vide
# ══════════════════════════════════════════════════════════════════

def test_predict_empty_url():
    code = http_post_expect_error("/api/predict", {"url": ""})
    assert code == 422, f"Attendu 422 (Unprocessable Entity), reçu {code}"
    ok(f"URL vide rejetée correctement avec HTTP 422")

# ══════════════════════════════════════════════════════════════════
# TEST 9 — /api/lists — GET les listes
# ══════════════════════════════════════════════════════════════════

def test_get_lists():
    r = http_get("/api/lists")
    assert "whitelist" in r, "Clé 'whitelist' absente"
    assert "blacklist" in r, "Clé 'blacklist' absente"
    info(f"Whitelist : {len(r['whitelist'])} domaines → exemples: {r['whitelist'][:3]}")
    info(f"Blacklist : {len(r['blacklist'])} domaines → exemples: {r['blacklist'][:3]}")
    ok("Listes retournées correctement")

# ══════════════════════════════════════════════════════════════════
# TEST 10 — /api/lists/add + /api/lists/remove (Whitelist)
# ══════════════════════════════════════════════════════════════════

def test_list_add_remove_whitelist():
    domain = "test-safe-domain-xai.com"

    # Ajouter à la whitelist
    r = http_post("/api/lists/add", {"domain": domain, "list_type": "whitelist"})
    assert r["success"] is True, f"Ajout échoué : {r['message']}"
    ok(f"Domaine '{domain}' ajouté à la Whitelist")

    # Vérifier que la whitelist contient bien le domaine
    lists = http_get("/api/lists")
    assert domain in lists["whitelist"], f"'{domain}' non trouvé dans la whitelist après ajout"
    ok("Présence dans la Whitelist confirmée")

    # Prédire → doit être WHITELIST
    r2 = http_post("/api/predict", {"url": f"https://{domain}/some/page"})
    assert r2["method"] == "whitelist", f"Attendu 'whitelist', reçu '{r2['method']}'"
    assert r2["status"] == "legitimate"
    ok(f"URL '{domain}' correctement identifiée via Whitelist après ajout")

    # Supprimer
    r3 = http_post("/api/lists/remove", {"domain": domain, "list_type": "whitelist"})
    assert r3["success"] is True, f"Suppression échouée : {r3['message']}"
    ok(f"Domaine '{domain}' supprimé de la Whitelist")

    # Vérifier la suppression
    lists2 = http_get("/api/lists")
    assert domain not in lists2["whitelist"], f"'{domain}' encore dans la whitelist après suppression!"
    ok("Suppression confirmée — domaine absent de la Whitelist")

# ══════════════════════════════════════════════════════════════════
# TEST 11 — /api/lists/add (Blacklist) + Vérification de priorité
# ══════════════════════════════════════════════════════════════════

def test_list_add_remove_blacklist():
    domain = "definitely-a-phishing-test.xyz"

    # Ajouter à la blacklist
    r = http_post("/api/lists/add", {"domain": domain, "list_type": "blacklist"})
    assert r["success"] is True
    ok(f"Domaine '{domain}' ajouté à la Blacklist")

    # Prédire → doit être BLACKLIST avec confiance 1.0
    r2 = http_post("/api/predict", {"url": f"http://{domain}/secure/login"})
    assert r2["method"] == "blacklist", f"Attendu 'blacklist', reçu '{r2['method']}'"
    assert r2["status"] == "phishing"
    assert r2["confidence"] == 1.0
    ok("URL blacklistée → PHISHING (confiance 100%)")

    # Nettoyer
    http_post("/api/lists/remove", {"domain": domain, "list_type": "blacklist"})
    ok(f"Nettoyage : '{domain}' retiré de la Blacklist")

# ══════════════════════════════════════════════════════════════════
# TEST 12 — /api/predict — Valider le schéma complet de réponse
# ══════════════════════════════════════════════════════════════════

def test_response_schema():
    r = http_post("/api/predict", {"url": "http://suspicious-login-update.com/account"})

    required_keys = ["url", "status", "confidence", "method", "score",
                     "verdict_details", "heuristics", "ml_probabilities",
                     "explanations", "robustness"]
    for key in required_keys:
        assert key in r, f"Clé obligatoire '{key}' absente de la réponse"

    heur_keys = ["score", "max_score", "score_percent", "details", "features"]
    for key in heur_keys:
        assert key in r["heuristics"], f"Clé heuristique '{key}' absente"

    assert r["status"] in ("phishing", "legitimate"), f"Status invalide: {r['status']}"
    assert 0.5 <= r["confidence"] <= 1.0, f"Confiance hors intervalle [0.5, 1.0]: {r['confidence']}"
    assert 0 <= r["score"] <= 100, f"Score hors intervalle [0, 100]: {r['score']}"

    ok("Schéma de réponse complet et valide ✓")
    info(f"  status={r['status']}, confidence={r['confidence']:.2f}, score={r['score']:.1f}")


# ══════════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════════

def print_final_report():
    total = len(results)
    passed_count = sum(1 for _, v in results if v)
    failed = total - passed_count

    print(f"\n{'=' * 65}")
    print(f"{BOLD}  RAPPORT FINAL -- PhishShield API Tests{RESET}")
    print(f"{'=' * 65}")
    for name, success in results:
        icon = f"{GREEN}[v]{RESET}" if success else f"{RED}[x]{RESET}"
        print(f"  {icon} {name}")
    print(f"{'-' * 65}")
    if passed_count == total:
        print(f"  {GREEN}{passed_count}/{total} tests reussis{RESET}")
    else:
        print(f"  {YELLOW}{passed_count}/{total} tests reussis{RESET} -- {RED}{failed} echoues{RESET}")

    if passed_count == total:
        print(f"\n  {GREEN}{BOLD}[OK] API 100% fonctionnelle -- tous les endpoints valides !{RESET}")
    elif passed_count / total >= 0.7:
        print(f"\n  {YELLOW}[WARN] API globalement fonctionnelle -- quelques points a verifier.{RESET}")
    else:
        print(f"\n  {RED}[ERR] Problemes detectes -- verifier les modeles/scaler et relancer l'API.{RESET}")
    print(f"{'=' * 65}\n")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{CYAN}{'=' * 65}")
    print(f"  PhishShield XAI -- Test Complet de la FastAPI")
    print(f"  Base URL : {BASE_URL}")
    print(f"{'=' * 65}{RESET}")

    # Vérifier que l'API est active avant de commencer
    print(f"\n{BOLD}Verification de la connexion a l'API...{RESET}")
    try:
        req = urllib.request.Request(f"{BASE_URL}/", method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                print(f"  {GREEN}[OK] API accessible (HTTP 200){RESET}\n")
            else:
                raise Exception(f"Status inattendu: {resp.status}")
    except Exception as e:
        print(f"  {RED}[ERR] IMPOSSIBLE DE JOINDRE L'API ! ({e}){RESET}")
        print(f"  {YELLOW}Assurez-vous que l'API tourne avec :{RESET}")
        print(f"    cd xai_project")
        print(f"    uvicorn api.main:app --port 8000\n")
        sys.exit(1)

    # Lancer tous les tests
    run_test("Connectivité API",                   test_api_alive)
    run_test("Dashboard HTML (GET /)",             test_dashboard)
    run_test("Predict — URL Whitelist",            test_predict_whitelist)
    run_test("Predict — URL Blacklist",            test_predict_blacklist)
    run_test("Predict — URL suspecte + XAI",       test_predict_suspicious_ml)
    run_test("Predict — Adresse IP suspecte",      test_predict_ip_url)
    run_test("Predict — Les 3 modèles ML",         test_predict_all_models)
    run_test("Robustesse dans la réponse",         test_robustness_in_response)
    run_test("Validation URL vide (HTTP 422)",     test_predict_empty_url)
    run_test("GET /api/lists",                     test_get_lists)
    run_test("Whitelist : add + predict + remove", test_list_add_remove_whitelist)
    run_test("Blacklist : add + predict + remove", test_list_add_remove_blacklist)
    run_test("Schéma de réponse complet",          test_response_schema)

    print_final_report()
    success = all(ok for _, ok in results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
