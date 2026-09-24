"""
95% CIs on per-method decision accuracy, from a single mixed-effects logistic
regression fit jointly across all four methods:

    correct ~ 0 + condition + trial_num_c + (1 | participant) + (1 | text_id)

- condition: fixed effect, one dummy per method (humanRL, LLM_decision,
  preferenceRL, rlvr) with no intercept, so each coefficient IS that method's
  log-odds of a correct human decision at the average trial number.
- trial_num_c: fixed effect, trial number (centered) to control for learning/
  fatigue effects over the course of the session.
- participant: random intercept. Participants are between-subjects (each
  person only did one method), so ids are namespaced by condition.
- text_id: random intercept. Texts ARE shared/crossed across methods (the
  same underlying text pool is reused with different explanation hints), so
  text_id is left as-is.

Fitting one joint model (rather than 4 separate ones) means all four
conditions share the same participant/text variance components, which is
the standard way to make the resulting CIs comparable to each other.
"""

import os
import sys
import sqlite3

# pymer4/rpy2 need R on the PATH/R_HOME, and a libstdc++ new enough for R's
# icu4c dependency (GLIBCXX_3.4.30+), *before the process starts* -- Quest's
# anaconda3 python bundles an older libstdc++ that gets loaded first and
# can't be swapped out via os.environ once the interpreter is already
# running. So on first run we re-exec ourselves with the right env set.
_R_HOME = (
    "/hpc/software/spack_v20d1/spack/opt/spack/linux-rhel7-x86_64/"
    "gcc-12.3.0/r-4.4.0-aaqsqjfnzw5p5c4twtddj4sxqi23pyfw/rlib/R"
)
_LIBSTDCXX = (
    "/hpc/software/spack_v20d1/spack/opt/spack/linux-rhel7-x86_64/gcc-10.4.0/"
    "gcc-12.3.0-s74p2j6ix3sux2sua2bxyx4v5akg7dtv/lib64/libstdc++.so.6"
)
if os.environ.get("_POWER_ANALYSIS_REEXEC") != "1":
    os.environ["_POWER_ANALYSIS_REEXEC"] = "1"
    os.environ["R_HOME"] = _R_HOME
    os.environ["PATH"] = os.path.join(_R_HOME, "bin") + os.pathsep + os.environ["PATH"]
    os.environ["LD_PRELOAD"] = _LIBSTDCXX
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymer4.models import Lmer
from scipy.special import expit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "statistic", "text_2026-09-23.db")

CONDITION_TABLES = {
    "humanRL": "rl_text_guess",
    "LLM_decision": "llm_decision_guess",
    "preferenceRL": "llm_preference_guess",
    "rlvr": "rlvr_guess",
}


def load_data():
    conn = sqlite3.connect(DB_PATH)
    frames = []
    for condition, table in CONDITION_TABLES.items():
        df = pd.read_sql(
            f"SELECT user_id, trial_num, text_id, correct FROM {table}", conn
        )
        df["condition"] = condition
        # Participants only ever did one condition (between-subjects), aside
        # from one shared test account; namespace by condition so the random
        # effect never conflates two different people/accounts.
        df["participant"] = condition + "_" + df["user_id"].astype(str)
        frames.append(df)
    conn.close()

    data = pd.concat(frames, ignore_index=True)
    data["correct"] = data["correct"].astype(int)
    data["condition"] = pd.Categorical(
        data["condition"], categories=list(CONDITION_TABLES.keys())
    )
    data["trial_num_c"] = data["trial_num"] - data["trial_num"].mean()
    return data


def fit_model(data):
    model = Lmer(
        "correct ~ 0 + condition + trial_num_c + (1 | participant) + (1 | text_id)",
        data=data,
        family="binomial",
    )
    model.fit(summary=True)
    return model


def summarize(model, data):
    coefs = model.coefs
    rows = []
    for condition in CONDITION_TABLES:
        term = f"condition{condition}"
        est = coefs.loc[term, "Estimate"]
        lo = coefs.loc[term, "2.5_ci"]
        hi = coefs.loc[term, "97.5_ci"]

        sub = data[data["condition"] == condition]
        rows.append(
            {
                "condition": condition,
                "n_trials": len(sub),
                "n_participants": sub["participant"].nunique(),
                "n_texts": sub["text_id"].nunique(),
                "raw_accuracy": sub["correct"].mean(),
                "model_accuracy": expit(est),
                "ci_low": expit(lo),
                "ci_high": expit(hi),
            }
        )
    return pd.DataFrame(rows)


def check_overlaps(summary):
    print("\nPairwise 95% CI overlap check:")
    any_overlap = False
    for i in range(len(summary)):
        for j in range(i + 1, len(summary)):
            a, b = summary.iloc[i], summary.iloc[j]
            overlap = not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])
            any_overlap = any_overlap or overlap
            flag = "OVERLAP" if overlap else "separated"
            print(f"  {a['condition']:>14} vs {b['condition']:<14}: {flag}")

    if any_overlap:
        print(
            "\n-> Some CIs overlap: current sample size can't reliably tell those "
            "methods apart yet -- collect more data."
        )
    else:
        print(
            "\n-> All CIs are separated: current sample size is enough to "
            "distinguish all four methods."
        )


def plot_forest(summary, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(summary))[::-1]
    err_low = summary["model_accuracy"] - summary["ci_low"]
    err_high = summary["ci_high"] - summary["model_accuracy"]

    ax.errorbar(
        summary["model_accuracy"],
        y_pos,
        xerr=[err_low, err_high],
        fmt="o",
        color="darkorange",
        ecolor="darkorange",
        capsize=5,
        markersize=8,
        linewidth=2,
        label="Model-estimated accuracy (95% CI)",
    )
    ax.scatter(
        summary["raw_accuracy"], y_pos, marker="x", color="black", zorder=5,
        label="Raw accuracy",
    )
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary["condition"])
    ax.set_xlabel("Accuracy (probability of a correct human decision)")
    ax.set_title(
        "Estimated accuracy by method with 95% CI\n"
        "(logistic GLMM: fixed = condition + trial_num, "
        "random = participant + text)"
    )
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved plot to {out_path}")


def main():
    data = load_data()
    print(f"Loaded {len(data)} trials across {data['condition'].nunique()} conditions")

    model = fit_model(data)

    # ================= 新增代码 =================
    print("\n=== 模型的 Coefficients (Log-odds) 与显著性检验 ===")
    # model.coefs 是一个 Pandas DataFrame，包含了 Estimate, SE, p-value 等
    print(model.coefs.to_string()) 
    print("===================================================\n")
    # ============================================

    summary = summarize(model, data)
    print("\n=== Accuracy estimates with 95% CI (from GLMM) ===")
    print(summary.to_string(index=False))

    csv_path = os.path.join(SCRIPT_DIR, "power_analysis_four_method_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary table to {csv_path}")

    check_overlaps(summary)
    plot_forest(summary, os.path.join(SCRIPT_DIR, "power_analysis_four_method.png"))


if __name__ == "__main__":
    main()
