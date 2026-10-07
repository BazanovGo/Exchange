"""Добавляет в леджер HAC-p (Newey–West) — у него нет дискретного порога 1/(B+1)."""
import pickle, logging, warnings
logging.disable(logging.INFO); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats as sps
from run_all import split
from exchange.stats.bootstrap import block_bootstrap

R = pickle.load(open("results.pkl", "rb"))
S = pickle.load(open("stats.pkl", "rb"))
T = S["table"].set_index(S["table"]["family"] + "|" + S["table"]["market"])
search = pd.read_pickle("search.pkl")


def hac_p(x):
    """Односторонний p для H0: E[r] <= 0 с ядром Бартлетта (Newey–West)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = x.size
    if n < 50:
        return np.nan
    d = x - x.mean()
    lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    gamma0 = float(d @ d) / n
    var = gamma0
    for k in range(1, lag + 1):
        g = float(d[k:] @ d[:-k]) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * g
    if var <= 0:
        return np.nan
    t = x.mean() / np.sqrt(var / n)
    return float(sps.norm.sf(t))


p_is, p_oos, b_oos = [], [], []
for _, row in search.iterrows():
    m = R["matrices"][row["key"]]
    is_idx, oos_idx = split(m.index)
    s = m[row["config"]]
    r_is, r_oos = s.loc[is_idx].dropna(), s.loc[oos_idx].dropna()
    p_is.append(hac_p(r_is)); p_oos.append(hac_p(r_oos))
    if row["winner"]:
        arr = np.asarray(r_oos, float); arr = arr[np.isfinite(arr)]
        sd = lambda v: float(np.mean(v) / np.std(v, ddof=1))
        b_oos.append(block_bootstrap(arr, sd, n_resamples=4000, seed=20261007)
                     .p_value_greater_than(0.0))
    else:
        b_oos.append(np.nan)

search["p_is_hac"] = p_is
search["p_oos_hac"] = p_oos
search["p_oos_boot4k"] = b_oos
search.to_pickle("search.pkl")
print(search[["key", "config", "winner", "is_sharpe", "oos_sharpe",
              "p_is", "p_is_hac", "p_oos", "p_oos_hac", "p_oos_boot4k"]]
      [search["winner"]].to_string(index=False))
print("min p_is_hac:", search["p_is_hac"].min())
