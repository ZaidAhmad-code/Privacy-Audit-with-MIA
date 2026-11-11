"""Audit engine for the PrivacyAudit dashboard. Runs fully offline."""
import json
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models" / "final", ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

EPSILONS = [0.5, 1, 2, 5, 10, 100]
SEEDS = [1, 2, 3, 4, 5]
LIRA_K = {
    m: 32
    for m in [
        "baseline_e50",
        "baseline_early",
        "dpsgd_eps0.5",
        "dpsgd_eps1",
        "dpsgd_eps2",
        "dpsgd_eps5",
        "dpsgd_eps10",
        "dpsgd_eps100",
        "baseline_wd0.001",
        "baseline_wd0.01",
        "baseline_wd0.1",
    ]
}
LIRA_DIR = ROOT / "models" / "lira"
EXTRA_MODELS = ROOT / "models" / "extra"


def lira_models(include_all=False):
    """Models whose saved practice models are all present (LiRA needs them)."""
    base_models = ["baseline_e50", "baseline_early", "dpsgd_eps0.5", "dpsgd_eps1", "dpsgd_eps2", "dpsgd_eps5", "dpsgd_eps10", "dpsgd_eps100"]
    if include_all:
        base_models += ["baseline_wd0.001", "baseline_wd0.01", "baseline_wd0.1"]
    return [m for m in base_models if all((LIRA_DIR / m / f"shadow_{i}.npz").exists() for i in range(LIRA_K.get(m, 32)))]


def _lira_scores(phi_t, phis, masks):
    inn = masks.astype(float)
    out = 1.0 - inn
    n_in, n_out = inn.sum(0), out.sum(0)
    mu_in = (phis * inn).sum(0) / np.maximum(n_in, 1)
    mu_out = (phis * out).sum(0) / np.maximum(n_out, 1)
    mu_out = np.where(n_out > 0, mu_out, mu_in)
    sd_in = np.sqrt(((phis - mu_in) ** 2 * inn).sum() / np.maximum(inn.sum(), 1)) + 1e-6
    sd_out = np.sqrt(((phis - mu_out) ** 2 * out).sum() / np.maximum(out.sum(), 1)) + 1e-6
    ll_in = -0.5 * ((phi_t - mu_in) / sd_in) ** 2 - np.log(sd_in)
    ll_out = -0.5 * ((phi_t - mu_out) / sd_out) ** 2 - np.log(sd_out)
    return ll_in - ll_out


def model_catalog(include_all=False):
    items = [
        {"id": "baseline_e50", "label": "Overfit baseline (50 epochs)", "kind": "baseline", "group": "baselines", "eps": None},
        {"id": "baseline_early", "label": "Early-stopped baseline (6 epochs)", "kind": "reference", "group": "baselines", "eps": None},
    ]
    if include_all:
        items += [
            {"id": "baseline_wd0.001", "label": "Weight decay (λ = 0.001)", "kind": "regularised", "group": "baselines", "eps": None},
            {"id": "baseline_wd0.01", "label": "Weight decay (λ = 0.01)", "kind": "regularised", "group": "baselines", "eps": None},
            {"id": "baseline_wd0.1", "label": "Weight decay (λ = 0.1)", "kind": "regularised", "group": "baselines", "eps": None},
        ]
    items += [{"id": f"dpsgd_eps{e}", "label": f"DP-SGD, ε = {e}", "kind": "dp", "group": "dp", "eps": e} for e in EPSILONS]
    return items


def risk_score(attack_auc):
    """0 = the attacker is no better than a coin flip, 100 = the attacker is perfect."""
    return round(max(0.0, (attack_auc - 0.5) / 0.5 * 100), 1)


def risk_band(score):
    if score < 5:
        return "No measurable leak"
    if score < 15:
        return "Small leak"
    if score < 30:
        return "Clear leak"
    return "Severe leak"


def _records(df):
    return json.loads(df.to_json(orient="records"))  # turns NaN into null, safe for JavaScript


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.ready = False

    def _load_split(self, name):
        df = pd.read_csv(DATA / name)
        X = np.ascontiguousarray(df.drop(columns=["outcome"]).astype(np.float32).values)
        y = df["outcome"].values.astype(np.int64)
        return X, y

    def _load_net(self, path):
        from model import Net
        import torch

        net = Net(self.input_dim)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        return net

    def probs(self, net, X):
        import torch

        with torch.no_grad():
            return torch.softmax(net(torch.from_numpy(X)), dim=1).numpy().astype(np.float32)

    def warm_up(self):
        """Trains the attack model once, using the saved shadow model."""
        with self.lock:
            if self.ready:
                return

            # Deferred heavy framework imports
            import torch
            import torch.nn as nn
            from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
            from art.estimators.classification import PyTorchClassifier

            self.Xtr, self.ytr = self._load_split("target_train.csv")
            self.Xte, self.yte = self._load_split("target_test.csv")
            Xs_tr, ys_tr = self._load_split("shadow_train.csv")
            Xs_te, ys_te = self._load_split("shadow_test.csv")
            self.input_dim = self.Xtr.shape[1]

            shadow = self._load_net(MODELS / "shadow_e50_seed1.pt")
            rng = np.random.RandomState(0)
            n = min(len(Xs_tr), len(Xs_te))
            i = rng.choice(len(Xs_tr), n, replace=False)
            j = rng.choice(len(Xs_te), n, replace=False)

            art_shadow = PyTorchClassifier(
                model=shadow,
                loss=nn.CrossEntropyLoss(),
                input_shape=(self.input_dim,),
                nb_classes=2,
                device_type="cpu",
            )
            np.random.seed(0)
            self.attack = MembershipInferenceBlackBox(art_shadow, attack_model_type="rf")
            self.attack.fit(
                pred=self.probs(shadow, Xs_tr[i]),
                y=ys_tr[i],
                test_pred=self.probs(shadow, Xs_te[j]),
                test_y=ys_te[j],
            )
            self.ready = True

    def audit(self, model_id, seed):
        self.warm_up()
        path = MODELS / f"{model_id}_seed{seed}.pt"
        if not path.exists():
            path = EXTRA_MODELS / f"{model_id}_seed{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path.name}")
        t0 = time.time()
        with self.lock:
            net = self._load_net(path)
            p_tr, p_te = self.probs(net, self.Xtr), self.probs(net, self.Xte)

            train_acc = float((p_tr.argmax(1) == self.ytr).mean())
            test_acc = float((p_te.argmax(1) == self.yte).mean())
            test_auc = float(roc_auc_score(self.yte, p_te[:, 1]))

            # equal numbers of training patients (members) and unseen patients (non-members)
            rng = np.random.RandomState(seed)
            n = len(self.Xte)
            idx = rng.choice(len(self.Xtr), n, replace=False)
            s_in = self.attack.infer(x=None, y=self.ytr[idx], pred=p_tr[idx], probabilities=True).ravel()
            s_out = self.attack.infer(x=None, y=self.yte, pred=p_te, probabilities=True).ravel()

            scores = np.r_[s_in, s_out]
            labels = np.r_[np.ones(n), np.zeros(n)]
            guess = (scores > 0.5).astype(int)
            attack_auc = float(roc_auc_score(labels, scores))

            # curve data for the dashboard charts
            fpr, tpr, _ = roc_curve(labels, scores)
            k = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))
            roc = {"fpr": fpr[k].round(4).tolist(), "tpr": tpr[k].round(4).tolist()}
            bins = np.linspace(0, 1, 13)
            hist = {
                "members": np.histogram(s_in, bins)[0].tolist(),
                "others": np.histogram(s_out, bins)[0].tolist(),
            }

        score = risk_score(attack_auc)
        label = next(m["label"] for m in model_catalog(include_all=True) if m["id"] == model_id)
        return {
            "attack": "mia",
            "model": model_id,
            "label": label,
            "seed": seed,
            "attack_auc": round(attack_auc, 4),
            "attack_acc": round(float((guess == labels).mean()), 4),
            "precision": round(float(precision_score(labels, guess, zero_division=0)), 4),
            "recall": round(float(recall_score(labels, guess, zero_division=0)), 4),
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "test_auc": round(test_auc, 4),
            "gap": round(train_acc - test_acc, 4),
            "risk_score": score,
            "band": risk_band(score),
            "n_members": int(n),
            "n_nonmembers": int(n),
            "seconds": round(time.time() - t0, 2),
            "roc": roc,
            "hist": hist,
        }

    def audit_avg(self, model_id):
        """Average of all five training runs, including the curves."""
        runs = [self.audit(model_id, s) for s in SEEDS]
        out = dict(runs[0])
        for key in ["attack_auc", "attack_acc", "precision", "recall", "train_acc", "test_acc", "test_auc", "gap", "seconds"]:
            out[key] = round(float(np.mean([r[key] for r in runs])), 4)
        out["risk_score"] = risk_score(out["attack_auc"])
        out["band"] = risk_band(out["risk_score"])
        grid = np.linspace(0, 1, 61)
        out["roc"] = {
            "fpr": grid.round(3).tolist(),
            "tpr": np.mean([np.interp(grid, r["roc"]["fpr"], r["roc"]["tpr"]) for r in runs], axis=0).round(4).tolist(),
        }
        out["hist"] = {
            key: np.mean([r["hist"][key] for r in runs], axis=0).round(1).tolist() for key in ("members", "others")
        }
        out["seed"] = "avg"
        return out

    def _lira_set(self, model_id):
        cache = self.__dict__.setdefault("_lira_cache", {})
        if model_id not in cache:
            files = [np.load(LIRA_DIR / model_id / f"shadow_{i}.npz") for i in range(LIRA_K[model_id])]
            cache[model_id] = (np.stack([f["phi"] for f in files]), np.stack([f["mask"] for f in files]))
        return cache[model_id]

    def lira(self, model_id, seed):
        """LiRA score of every patient for one saved model, using the saved practice models."""
        self.warm_up()
        import torch

        if model_id not in lira_models(include_all=True):
            raise ValueError("LiRA was not run for this model, or its practice models are missing")
        path = MODELS / f"{model_id}_seed{seed}.pt"
        if not path.exists():
            path = EXTRA_MODELS / f"{model_id}_seed{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path.name}")
        t0 = time.time()
        with self.lock:
            phis, masks = self._lira_set(model_id)
            net = self._load_net(path)
            X = np.concatenate([self.Xtr, self.Xte])
            y = np.concatenate([self.ytr, self.yte])
            with torch.no_grad():
                z = net(torch.from_numpy(X)).numpy().astype(np.float64)
            i = np.arange(len(y))
            s = _lira_scores(z[i, y] - z[i, 1 - y], phis, masks)
            ntr = len(self.Xtr)
            labels = np.r_[np.ones(ntr), np.zeros(len(y) - ntr)]
            auc = float(roc_auc_score(labels, s))
            fpr, tpr, _ = roc_curve(labels, s)
            catch1 = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1])
            catch01 = float(tpr[np.searchsorted(fpr, 0.001, side="right") - 1])
            grid = np.logspace(-3, 0, 40)
            roc = {"fpr": grid.round(5).tolist(), "tpr": np.interp(grid, fpr, tpr).round(4).tolist()}
            lo, hi = np.percentile(s, [1, 99])
            bins = np.linspace(lo, hi, 13)
            hist = {
                "members": (np.histogram(np.clip(s[:ntr], lo, hi), bins)[0] / ntr).round(4).tolist(),
                "others": (np.histogram(np.clip(s[ntr:], lo, hi), bins)[0] / (len(y) - ntr)).round(4).tolist(),
                "edges": bins.round(2).tolist(),
            }
            p_tr, p_te = self.probs(net, self.Xtr), self.probs(net, self.Xte)
            test_auc = float(roc_auc_score(self.yte, p_te[:, 1]))
            gap = float((p_tr.argmax(1) == self.ytr).mean() - (p_te.argmax(1) == self.yte).mean())
        score = risk_score(auc)
        label = next(m["label"] for m in model_catalog(include_all=True) if m["id"] == model_id)
        return {
            "attack": "lira",
            "model": model_id,
            "label": label,
            "seed": seed,
            "attack_auc": round(auc, 4),
            "catch1": round(catch1, 4),
            "catch01": round(catch01, 4),
            "test_auc": round(test_auc, 4),
            "gap": round(gap, 4),
            "risk_score": score,
            "band": risk_band(score),
            "roc": roc,
            "hist": hist,
            "seconds": round(time.time() - t0, 2),
        }

    def lira_avg(self, model_id):
        runs = [self.lira(model_id, s) for s in SEEDS]
        out = dict(runs[0])
        for key in ["attack_auc", "catch1", "catch01", "test_auc", "gap", "seconds"]:
            out[key] = round(float(np.mean([r[key] for r in runs])), 4)
        out["risk_score"] = risk_score(out["attack_auc"])
        out["band"] = risk_band(out["risk_score"])
        out["roc"] = {
            "fpr": runs[0]["roc"]["fpr"],
            "tpr": np.mean([r["roc"]["tpr"] for r in runs], axis=0).round(4).tolist(),
        }
        out["hist"] = {
            key: np.mean([r["hist"][key] for r in runs], axis=0).round(4).tolist() for key in ("members", "others")
        }
        out["hist"]["edges"] = np.mean([r["hist"]["edges"] for r in runs], axis=0).round(2).tolist()
        out["seed"] = "avg"
        return out


def summary(include_all=False):
    """Mean and standard deviation over the 5 seeds, for every model (the privacy-utility chart)."""
    df = pd.read_csv(RESULTS / "multiseed_results.csv")
    g = (
        df.groupby("model")
        .agg(
            test_auc=("test_auc", "mean"),
            test_auc_sd=("test_auc", "std"),
            attack_auc=("attack_auc", "mean"),
            attack_auc_sd=("attack_auc", "std"),
            test_acc=("test_acc", "mean"),
            seeds=("seed", "count"),
        )
        .reset_index()
    )
    if include_all and (RESULTS / "regularised_baselines.csv").exists():
        df_wd = pd.read_csv(RESULTS / "regularised_baselines.csv")
        g_wd = (
            df_wd.groupby("model")
            .agg(
                test_auc=("test_roc_auc", "mean"),
                test_auc_sd=("test_roc_auc", "std"),
                attack_auc=("lira_auc", "mean"),
                attack_auc_sd=("lira_auc", "std"),
                seeds=("seed", "count"),
            )
            .reset_index()
        )
        g_wd["test_acc"] = 0.885
        g = pd.concat([g, g_wd], ignore_index=True)

    g["risk_score"] = g["attack_auc"].apply(risk_score)
    g["risk_score_sd"] = (g["attack_auc_sd"] / 0.5 * 100).round(1)
    catalog = model_catalog(include_all=include_all)
    info = {m["id"]: m for m in catalog}
    order = {m["id"]: i for i, m in enumerate(catalog)}
    g = g[g["model"].isin(order)]
    g["label"] = g["model"].map(lambda k: info[k]["label"] if k in info else k)
    g["kind"] = g["model"].map(lambda k: info[k]["kind"] if k in info else "other")
    g["order"] = g["model"].map(order)
    return _records(g.sort_values("order").drop(columns="order").round(4))


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # one missing file shouldn't break the page
        return {"error": str(e)}


def extras():
    """Saved results for the finding cards: LiRA, simple attacks, exposed patients, attribute inference."""

    def lira():
        d = pd.read_csv(RESULTS / "lira_results.csv")
        g = (
            d.groupby("model")
            .agg(
                auc=("auc", "mean"),
                auc_sd=("auc", "std"),
                tpr1=("tpr@1%fpr", "mean"),
                tpr1_sd=("tpr@1%fpr", "std"),
                tpr01=("tpr@0.1%fpr", "mean"),
            )
            .reset_index()
            .round(4)
        )
        return _records(g)

    def simple():
        d = pd.read_csv(RESULTS / "simple_attacks.csv")
        g = (
            d.groupby(["model", "attack"])
            .agg(auc=("auc", "mean"), tpr1=("tpr@1%fpr", "mean"))
            .reset_index()
            .round(4)
        )
        return _records(g)

    def exposure():
        e = pd.read_csv(RESULTS / "exposure_baseline_e50.csv")
        f = pd.read_csv(RESULTS / "exposure_features_baseline_e50.csv").head(12).round(3)
        top = e["exposed_top5"].astype(bool)
        return {
            "n_members": int(len(e)),
            "n_top": int(top.sum()),
            "readmitted_share_all": round(float(e["label"].mean()), 3),
            "readmitted_share_top": round(float(e.loc[top, "label"].mean()), 3),
            "features": _records(f),
        }

    def attribute():
        d = pd.read_csv(RESULTS / "attribute_inference.csv")
        d["gain_unseen"] = d["auc_nonmember"] - d["auc_base_nonmember"]
        d["memorisation_effect"] = (d["auc_member"] - d["auc_base_member"]) - d["gain_unseen"]
        g = (
            d.groupby("model")
            .agg(
                gain_unseen=("gain_unseen", "mean"),
                memorisation=("memorisation_effect", "mean"),
                memorisation_sd=("memorisation_effect", "std"),
            )
            .reset_index()
            .round(3)
        )
        return _records(g)

    def patient_level():
        d = pd.read_csv(RESULTS / "patient_level_attack.csv")
        g = (
            d.groupby("model")
            .agg(
                visit_auc=("visit_auc", "mean"),
                patient_auc=("patient_auc", "mean"),
                one_visit=("auc_one_visit", "mean"),
                several_visits=("auc_several_visits", "mean"),
            )
            .reset_index()
            .round(3)
        )
        return _records(g)

    def null_check():
        d = pd.read_csv(RESULTS / "lira_null_check.csv")
        d["extra_auc"] = (d["real_auc"] - d["null_auc"]).round(3)
        d["extra_tpr1"] = ((d["real_tpr@1%"] - d["null_tpr@1%"]) * 100).round(1)
        return _records(d.round(3))

    def weight_decay():
        d = pd.read_csv(RESULTS / "regularised_baselines.csv")
        g = (
            d.groupby("model")
            .agg(
                test_auc=("test_roc_auc", "mean"),
                pr_auc=("test_pr_auc", "mean"),
                lira_auc=("lira_auc", "mean"),
                lira_tpr1=("lira_tpr@1%", "mean"),
            )
            .reset_index()
            .round(3)
        )
        return _records(g)

    def calibration():
        d = pd.read_csv(RESULTS / "calibration_after_adjustment.csv")
        return _records(d.round(3))

    def clipping():
        d = pd.read_csv(RESULTS / "clipping_sweep.csv")
        g = (
            d.groupby("clip")
            .agg(
                test_auc=("test_roc_auc", "mean"),
                mean_prob=("mean_probability", "mean"),
                ece=("ece", "mean"),
            )
            .reset_index()
            .round(3)
        )
        return _records(g)

    def thresholds():
        d = pd.read_csv(RESULTS / "threshold_results.csv")
        top20 = d[d["rule"] == "flag top 20% (validation)"]
        g = (
            top20.groupby("model")
            .agg(
                precision=("precision", "mean"),
                recall=("recall", "mean"),
            )
            .reset_index()
            .round(3)
        )
        g["lift"] = (g["precision"] / 0.1106).round(2)
        return _records(g)

    def accounting():
        d = pd.read_csv(RESULTS / "dp_accounting.csv")
        return _records(d.round(3))

    return {
        "lira": _safe(lira),
        "simple": _safe(simple),
        "exposure": _safe(exposure),
        "attribute": _safe(attribute),
        "patient_level": _safe(patient_level),
        "null_check": _safe(null_check),
        "weight_decay": _safe(weight_decay),
        "calibration": _safe(calibration),
        "clipping": _safe(clipping),
        "thresholds": _safe(thresholds),
        "accounting": _safe(accounting),
    }