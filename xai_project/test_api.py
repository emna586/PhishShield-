"""
Script de test de l'API FastAPI PhishShield.
Teste les endpoints principaux avec des URLs réelles.
"""
import urllib.request
import json
import sys


def call_api(url_to_test, endpoint="http://127.0.0.1:8000/api/predict"):
    """Envoie une requête POST à l'API et retourne la réponse JSON."""
    data = json.dumps({"url": url_to_test}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def print_result(label, r):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  URL       : {r['url']}")
    print(f"  Verdict   : {r['status'].upper()}")
    print(f"  Methode   : {r['method']}")
    print(f"  Score     : {r['score']:.1f}/100")
    print(f"  Confiance : {r['confidence']*100:.1f}%")
    print(f"  Details   : {r['verdict_details']}")
    ml = r.get("ml_probabilities", {})
    if ml:
        print(f"  ML Probas : {', '.join(f'{k}: {v*100:.1f}%' for k, v in ml.items())}")
    expl = r.get("explanations")
    if expl and expl.get("shap"):
        top = expl["shap"][0]
        print(f"  SHAP top1 : {top['feature']} ({top['effect']}, imp={top['importance']:.4f})")


def main():
    tests = [
        ("google.com [WHITELIST - LEGITIMATE]",
         "https://www.google.com"),
        ("phishing-bank.com [BLACKLIST - PHISHING]",
         "http://phishing-bank.com/secure/login"),
        ("IP suspecte [SUSPICIOUS]",
         "http://123.45.67.89/update?id=992&bank=paypal"),
        ("Sous-domaines suspects [HEURISTICS]",
         "http://secure-verify.paypal-account.update-now.com/login"),
        ("HTTPS normal [LEGITIMATE]",
         "https://github.com/python/cpython"),
    ]

    print("\nPhishShield API — Tests en direct")
    print("="*60)

    passed = 0
    for label, url in tests:
        try:
            result = call_api(url)
            print_result(label, result)
            passed += 1
        except Exception as e:
            print(f"\n[ERREUR] {label}: {e}")

    print(f"\n{'='*60}")
    print(f"RESULTATS: {passed}/{len(tests)} tests reussis")
    if passed == len(tests):
        print("Tous les tests ont reussi!")
    print("="*60)
    return passed == len(tests)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
