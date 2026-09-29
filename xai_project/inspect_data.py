import json
from pathlib import Path

out = Path("outputs/validation")

def load(f):
    p = out / f
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

full      = load("full_metrics_report.json")
ml_dl     = load("ml_vs_dl_metrics.json")
bs        = load("bootstrap_report.json")
anova     = load("anova_report.json")
native_ml = load("per_dataset_native_metrics.json")
native_dl = load("per_dataset_dl_metrics.json")

print("LOADED OK")
for k, v in full.items():
    print(f"  full[{k}]: {list(v.keys()) if isinstance(v, dict) else v}")
for k, v in ml_dl.items():
    print(f"  ml_dl[{k}]: {list(v.keys()) if isinstance(v, dict) else v}")
print("bootstrap:", list(bs.keys()))
print("anova:", list(anova.keys()))
print("native_ml datasets:", list(native_ml.keys()))
print("native_dl datasets:", list(native_dl.keys()))
