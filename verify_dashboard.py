"""Checks that the dashboard shows what the experiments actually produced.

Put this file in the main project folder (next to dashboard/, data/, models/, results/) and run:
    python verify_dashboard.py
It needs no browser. It talks to the dashboard code directly and compares it with your saved results.
"""
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "dashboard"))

passes, fails = 0, []


def check(name, cond, detail=""):
    global passes
    if cond:
        passes += 1
        print(f"  PASS  {name}")
    else:
        fails.append(name)
        print(f"  FAIL  {name}   {detail}")


def section(title):
    def deco(fn):
        print(f"\n== {title} ==")
        try:
            fn()
        except Exception as e:                       # a crash is reported as a failure, not hidden
            fails.append(f"{title}: crashed")
            print("  FAIL  this section crashed:", repr(e))
            traceback.print_exc(limit=3)
        return fn
    return deco


def _bad_constant(c):
    raise ValueError(f"invalid JSON value {c}")      # NaN or Infinity would break the page


S = {}   # shared state between sections


@section("1. Files are in place")
def _():
    import audit
    S["audit"] = audit
    S["ids"] = [m["id"] for m in audit.model_catalog()]
    ids, seeds = S["ids"], audit.SEEDS
    need = {
        "data": ["target_train.csv", "target_test.csv", "shadow_train.csv", "shadow_test.csv"],
        "results": ["multiseed_results.csv", "simple_attacks.csv", "lira_results.csv", "attribute_inference.csv",
                    "exposure_baseline_e50.csv", "exposure_features_baseline_e50.csv"],
        "models/final": [f"{m}_seed{s}.pt" for m in ids for s in seeds] + [f"{m}_seed{s}.json" for m in ids for s in seeds]
                        + ["shadow_e50_seed1.pt"],
    }
    for folder, files in need.items():
        miss = [f for f in files if not (ROOT / folder / f).exists()]
        check(f"{folder}: {len(files) - len(miss)} of {len(files)} files present", not miss, f"missing, for example {miss[:3]}")
    check("font file present", (ROOT / "dashboard/static/fonts/geist-latin-wght-normal.woff2").exists())


@section("2. No leftovers from the old version")
def _():
    for f in ["dashboard/audit.py", "dashboard/app.py", "dashboard/static/app.js", "dashboard/static/index.html"]:
        txt = (ROOT / f).read_text(encoding="utf-8")
        check(f"{f} has no old model name (baseline_early_e4)", "baseline_early_e4" not in txt)
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    py = (ROOT / "dashboard/app.py").read_text(encoding="utf-8")
    for route in sorted(set(re.findall(r"/api/[a-z]+", js))):
        check(f"app.js calls {route} and app.py serves it", route in py)


@section("3. Data and models belong together")
def _():
    audit, ids = S["audit"], S["ids"]
    d = {n: pd.read_csv(ROOT / "data" / f"{n}.csv") for n in ["target_train", "target_test", "shadow_train", "shadow_test"]}
    cols = [list(x.columns) for x in d.values()]
    check("all four data files have identical columns", all(c == cols[0] for c in cols))
    input_dim = d["target_train"].shape[1] - 1
    S["input_dim"] = input_dim
    metas = []
    for m in ids:
        for s in audit.SEEDS:
            p = ROOT / "models/final" / f"{m}_seed{s}.json"
            if p.exists():
                metas.append(json.load(open(p)))
    check(f"all {len(metas)} models were trained on the same number of rows as target_train.csv ({len(d['target_train'])})",
          all(x["n_train"] == len(d["target_train"]) for x in metas),
          f"rows in data: {len(d['target_train'])}, in models: {sorted({x['n_train'] for x in metas})}")
    check(f"all models expect the same number of features as the data ({input_dim})",
          all(x["input_dim"] == input_dim for x in metas))
    dp = [x for x in metas if x["kind"] == "dp-sgd"]
    check(f"all {len(dp)} DP models stayed inside their privacy budget (spent epsilon <= target)",
          all(x["spent_epsilon"] <= x["target_epsilon"] * 1.001 for x in dp))
    early = [x for x in metas if x["model"] == "baseline_early"]
    label = next(m["label"] for m in audit.model_catalog() if m["id"] == "baseline_early")
    check(f"dashboard label for the early-stopped model matches the epochs used ({early[0]['epochs']})",
          str(early[0]["epochs"]) in label, f"label says: {label}")


@section("4. Result files are complete")
def _():
    audit, ids = S["audit"], S["ids"]
    ms = pd.read_csv(ROOT / "results/multiseed_results.csv")
    S["ms"] = ms
    check("multiseed_results.csv has 5 seeds x 8 models = 40 rows", len(ms) == 40, f"has {len(ms)} rows")
    check("no missing values in test_auc, test_acc, attack_auc", not ms[["test_auc", "test_acc", "attack_auc"]].isna().any().any())
    check("model names in the results match the dashboard's list", set(ms["model"]) == set(ids), f"results: {sorted(set(ms['model']))}")
    lr = pd.read_csv(ROOT / "results/lira_results.csv")
    S["lr"] = lr
    check("lira_results.csv has exactly 5 rows (seeds) for each model", bool(lr.groupby("model").size().eq(5).all()),
          str(lr.groupby("model").size().to_dict()))
    check("simple_attacks.csv has 120 rows", len(pd.read_csv(ROOT / "results/simple_attacks.csv")) == 120)
    check("attribute_inference.csv has 40 rows", len(pd.read_csv(ROOT / "results/attribute_inference.csv")) == 40)


@section("5. The live audit matches the experiment")
def _():
    import app as appmod
    # Modified to call get_engine() so app.py doesn't need modification
    S["engine"], S["client"] = appmod.get_engine(), appmod.app.test_client()
    engine, ms, ids, audit = S["engine"], S["ms"], S["ids"], S["audit"]
    engine.warm_up()
    bad, gaps = [], []
    for m in ids:
        for s in audit.SEEDS:
            r = engine.audit(m, s)
            row = ms[(ms["model"] == m) & (ms["seed"] == s)].iloc[0]
            for k in ("test_auc", "test_acc", "train_acc"):
                if abs(r[k] - row[k]) > 2e-4:
                    bad.append((m, s, k, r[k], round(float(row[k]), 4)))
            gaps.append(abs(r["attack_auc"] - row["attack_auc"]))
            S.setdefault("audits", {})[(m, s)] = r
    check("test AUC, test accuracy and train accuracy match the experiment for all 40 models", not bad, str(bad[:2]))
    check(f"live attack score within 0.04 of the experiment (largest gap {max(gaps):.3f}, average {np.mean(gaps):.3f})",
          max(gaps) < 0.04, "the dashboard trains its own attack model, so small gaps are normal; large ones are not")


@section("6. The LiRA view matches the LiRA results")
def _():
    engine, lr, audit = S["engine"], S["lr"], S["audit"]
    avail = audit.lira_models()
    expected = ["baseline_e50", "baseline_early", "dpsgd_eps0.5", "dpsgd_eps1", "dpsgd_eps2", "dpsgd_eps5", "dpsgd_eps10", "dpsgd_eps100"]
    check("practice models are present for all 5 LiRA models", set(avail) == set(expected), f"available: {avail}")
    bad = []
    for m in avail:
        for s in audit.SEEDS:
            r = engine.lira(m, s)
            S.setdefault("liras", {})[(m, s)] = r
            row = lr[(lr["model"] == m) & (lr["seed"] == s)]
            if row.empty:
                bad.append((m, s, "no row in lira_results.csv"))
                continue
            row = row.iloc[0]
            if (abs(r["attack_auc"] - row["auc"]) > 5e-4 or abs(r["catch1"] - row["tpr@1%fpr"]) > 2e-3
                    or abs(r["catch01"] - row["tpr@0.1%fpr"]) > 2e-3):
                bad.append((m, s, round(r["attack_auc"], 4), round(float(row["auc"]), 4)))
    check(f"LiRA score and catch rates match lira_results.csv for all {len(avail) * 5} model and seed pairs", not bad,
          f"{bad[:2]}  (if this is the eps 100 model, check LIRA_K in audit.py matches the number of practice models you trained)")


@section("7. The web API returns the right numbers")
def _():
    client, audit, ms, lr = S["client"], S["audit"], S["ms"], S["lr"]

    def get(url):
        resp = client.get(url)
        return resp.status_code, json.loads(resp.get_data(as_text=True), parse_constant=_bad_constant)

    code, j = get("/api/models")
    check("/api/models lists 8 models", code == 200 and len(j["models"]) == 8)

    code, summ = get("/api/summary")
    mean = ms.groupby("model")[["test_auc", "attack_auc"]].mean()
    errs = []
    for row in summ:
        m = row["model"]
        if abs(row["test_auc"] - mean.loc[m, "test_auc"]) > 6e-4 or abs(row["attack_auc"] - mean.loc[m, "attack_auc"]) > 6e-4:
            errs.append(m)
        if abs(row["risk_score"] - max(0, (mean.loc[m, "attack_auc"] - 0.5) / 0.5 * 100)) > 0.15:
            errs.append(m + " risk")
    check("/api/summary equals the 5-seed averages in multiseed_results.csv, and risk scores follow the formula", code == 200 and not errs, str(errs))

    code, ex = get("/api/extras")
    lm = lr.groupby("model")["auc"].mean()
    ok = isinstance(ex["lira"], list) and all(abs(r["auc"] - lm[r["model"]]) < 6e-4 for r in ex["lira"])
    check("/api/extras LiRA table equals lira_results.csv", code == 200 and ok)
    e = pd.read_csv(ROOT / "results/exposure_baseline_e50.csv")
    top = e["exposed_top5"].astype(bool)
    check("exposed-patients share matches exposure_baseline_e50.csv",
          abs(ex["exposure"]["readmitted_share_top"] - e.loc[top, "label"].mean()) < 6e-4)

    code, a = get("/api/audit?model=baseline_e50&seed=avg")
    singles = [S["audits"][("baseline_e50", s)]["test_auc"] for s in audit.SEEDS]
    check("the 'Avg' result is the average of the five single runs", code == 200 and abs(a["test_auc"] - np.mean(singles)) < 6e-4)
    check("bad model name is refused", get("/api/audit?model=nope&seed=1")[0] == 400)
    check("bad run number is refused", get("/api/audit?model=baseline_e50&seed=9")[0] == 400)
    check("LiRA for a model it was not run on is refused", get("/api/lira?model=nope&seed=1")[0] != 200)

    probs = []
    for (m, s), r in list(S["audits"].items()) + list(S.get("liras", {}).items()):
        f, t = r["roc"]["fpr"], r["roc"]["tpr"]
        if not (np.all(np.diff(f) >= -1e-9) and np.all(np.diff(t) >= -1e-9) and 0 <= min(t) and max(t) <= 1.0001):
            probs.append((m, s, "roc"))
        if not 0 <= r["risk_score"] <= 100:
            probs.append((m, s, "score"))
        total = sum(r["hist"]["members"])
        want = r.get("n_members", 1)
        if abs(total - want) > (0.01 if r["attack"] == "lira" else 0.5):
            probs.append((m, s, f"hist {total} vs {want}"))
    check(f"curve data is sane for all {len(S['audits']) + len(S.get('liras', {}))} audits (ROC rises, histograms add up)", not probs, str(probs[:3]))


@section("8. What you should see on screen (compare by eye, 2 minutes)")
def _():
    audit = S["audit"]
    engine = S["engine"]
    code, summ = S["client"].get("/api/summary").status_code, json.loads(S["client"].get("/api/summary").get_data(as_text=True))
    by = {r["model"]: r for r in summ}
    dp = [r for r in summ if r["kind"] == "dp"]
    print("  Top tiles in the MIA view:")
    print(f"    Overfit model risk       {by['baseline_e50']['risk_score']:.1f}")
    print(f"    Worst DP risk            {max(r['risk_score'] for r in dp):.1f}")
    print(f"    Best DP usefulness       {max(r['test_auc'] for r in dp):.3f}")
    print(f"    Cost of privacy          {max(r['test_auc'] for r in dp) - by['dpsgd_eps0.5']['test_auc']:.3f}")
    for m, s in (("baseline_e50", 1), ("dpsgd_eps1", 1)):
        r = S["audits"][(m, s)]
        print(f"\n  MIA view, {r['label']}, run {s}:")
        print(f"    gauge {r['risk_score']} ({r['band']}) | attack {r['attack_auc']:.3f} | usefulness {r['test_auc']:.3f}"
              f" | attacker accuracy {r['attack_acc']:.3f} | gap {r['gap']:.3f}")
        if (m, s) in S.get("liras", {}):
            l = S["liras"][(m, s)]
            print(f"  LiRA view, same model:")
            print(f"    gauge {l['risk_score']} ({l['band']}) | attack {l['attack_auc']:.3f} | caught at 1% {l['catch1'] * 100:.1f}%"
                  f" | caught at 0.1% {l['catch01'] * 100:.1f}%")
    print("\n  Open the dashboard, pick those models and run 1, and compare these numbers with the screen.")


print("\n" + "=" * 60)
print(f"{passes} checks passed, {len(fails)} failed")
for f in fails:
    print("  FAILED:", f)
print("=" * 60)
sys.exit(1 if fails else 0)