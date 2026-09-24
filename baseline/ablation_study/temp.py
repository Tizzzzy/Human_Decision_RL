import json
import numpy as np
import pandas as pd
from scipy import stats

def p_chosen_better(kc, nc, kr, nr):
    """
    Calculates P(p_c > p_r) under independent Beta(1+k, 1+n-k) posteriors 
    using numerical integration.
    """
    grid = np.linspace(0, 1, 501)
    cdf_r = stats.beta.cdf(grid, 1 + kr, 1 + nr - kr)
    pdf_c = stats.beta.pdf(grid, 1 + kc, 1 + nc - kc)
    return np.trapezoid(pdf_c * cdf_r, grid)

def main():
    print("Loading data...")
    try:
        with open("human_preferred_explanations.json", "r") as f:
            hp = json.load(f)
        with open("preference_pairs_aggregated.json", "r") as f:
            mp = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the JSON files are in the same directory.")
        return

    print("Filtering ill-defined entries...")
    # Find entries to exclude (same-source pairs or mismatched pairs)
    mismatched, same_source_mp = [], []
    for key, v in mp.items():
        pair_mp = {v["chosen_source"]} | {r["source"] for r in v["rejected_explanations"]}
        if len(pair_mp) == 1:
            same_source_mp.append(key)
        
        if key in hp:
            h = hp[key]
            # Safely handle key variations
            h_chosen = h.get("chosen_explanation_source", h.get("chosen_source"))
            h_rejected = h.get("rejected_explanation_source", h.get("rejected_source"))
            pair_hp = {h_chosen, h_rejected}
            
            if pair_hp != pair_mp:
                mismatched.append(key)

    EXCLUDE = set(mismatched) | set(same_source_mp)

    print("Calculating posteriors and building dataframe...")
    rows = []
    for key, v in hp.items():
        if key in EXCLUDE:
            continue
            
        m = mp[key]
        
        # Extract human decision stats
        nc = v["chosen_decisions_count"]
        nr = v["rejected_decisions_count"]
        kc = int(round((v.get("chosen_accuracy") or 0) * nc))
        kr = int(round((v.get("rejected_accuracy") or 0) * nr))
        
        # Calculate posterior confidence
        conf = p_chosen_better(kc, nc, kr, nr)
        
        # Determine agreement and scope
        h_chosen_src = v.get("chosen_explanation_source", v.get("chosen_source"))
        agree = (m["chosen_source"] == h_chosen_src)
        both_arms = v["decision_method"].startswith("both")
        
        rows.append({
            "key": key,
            "conf": conf,
            "agree": agree,
            "both_arms": both_arms
        })

    df = pd.DataFrame(rows)

    # 1. Primary analysis: Filter for entries where humans saw both arms
    B = df[df["both_arms"]]

    # 2. Isolate disagreements
    disagreements = B[~B["agree"]]
    n_disagree = len(disagreements)

    # 3. Calculate "clearly higher" thresholds
    clear_80 = disagreements[disagreements["conf"] >= 0.80]
    clear_90 = disagreements[disagreements["conf"] >= 0.90]

    # Output Results
    print("\n--- RESULTS ---")
    print(f"Total entries with both arms observed: {len(B)}")
    print(f"Total LLM/Human disagreements: {n_disagree}")
    print("-" * 50)
    print(f"Of the disagreements, human accuracy is clearly higher (conf >= 0.80): {len(clear_80)} ({len(clear_80) / n_disagree:.1%})")
    print(f"Of the disagreements, human accuracy is clearly higher (conf >= 0.90): {len(clear_90)} ({len(clear_90) / n_disagree:.1%})")

if __name__ == "__main__":
    main()