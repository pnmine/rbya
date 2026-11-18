"""Episode-level (cluster) bootstrap for paired differences, plus per-episode win counts and an exact sign test."""
import sys, math
import numpy as np, pandas as pd
rng = np.random.default_rng(0)

def load(folder):
    d = pd.read_csv(f"{folder}/incidents.csv")
    d["ttr_min"] = pd.to_numeric(d["ttr_min"], errors="coerce")
    return d

def paired(d, a, b, col):
    """mean per-incident difference b - a, resampling whole episodes (all 16 paired incidents of each)."""
    k = ["episode", "index"]
    x = d[d.agent == a].set_index(k)[col]; y = d[d.agent == b].set_index(k)[col]
    diff = (y - x).dropna()
    eps = diff.index.get_level_values(0)
    groups = [diff[eps == e].values for e in sorted(set(eps))]
    boots = []
    for _ in range(4000):
        pick = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[i] for i in pick]); boots.append(v.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per_ep = [g.mean() for g in groups]
    wins = sum(m < 0 for m in per_ep); ties = sum(m == 0 for m in per_ep); n = len(per_ep) - ties
    # exact two-sided sign test on the episodes that are not ties
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(wins, n - wins) + 1)) / 2 ** n) if n else 1.0
    return diff.mean(), lo, hi, wins, ties, len(per_ep), p

def marginal_ci(d, arm, col):
    g = d[d.agent == arm]; eps = sorted(g.episode.unique())
    groups = [g[g.episode == e][col].dropna().values for e in eps]
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(4000)]
    return g[col].mean(), *np.percentile(boots, [2.5, 97.5])

if __name__ == "__main__":
    folder, a, b = sys.argv[1:4]
    d = load(folder)
    for col in ("unnecessary", "collateral_ticks", "ttr_min"):
        m, lo, hi, w, t, n, p = paired(d, a, b, col)
        print(f"{folder:32s} {b}-{a} {col:16s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  {b} better in {w}/{n} episodes ({t} ties), sign p={p:.4f}")
