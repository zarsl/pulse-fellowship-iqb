"""Shared helpers."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # repo root, where this file lives
DATA_V1 = ROOT / "data" / "v1"
EXT = ROOT / "data" / "external"  # raw downloads of external data (gitignored)
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
for _d in (EXT, FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)  # percentiles reported by M-Lab
Q_GRID = np.arange(1, 100)  # P1..P99 for the full-range curve
METRICS = ("download", "upload", "latency", "loss")
HIGHER_IS_BETTER = {"download": True, "upload": True, "latency": False, "loss": False}
REQ_TO_METRIC = {
    "download_throughput_mbps": "download",
    "upload_throughput_mbps": "upload",
    "latency_ms": "latency",
    "packet_loss": "loss",
}
METRIC_TO_REQ = {v: k for k, v in REQ_TO_METRIC.items()}
USE_CASES = (
    "web browsing",
    "video streaming",
    "audio streaming",
    "video conferencing",
    "online backup",
    "gaming",
)
# Reporting standard
TOL = 0.05  # a displayed score must be within ±TOL of the score
CONF = 0.95  # with at least this probability
DEFF = 3  # design effect: n clustered tests carry about the information of n / DEFF independent ones
Z = 1.959964  # two-sided 95 % normal quantile

# Data sufficiency: every pass rate known to within ±TOL at 95 % confidence (worst case, pass rate 0.5)
MIN_TESTS = int(
    np.ceil(Z**2 * 0.25 / TOL**2 * DEFF)
)  # = 1,153 tests per reporting period
FLOOR_TESTS = 100  # below this, P1 and P99 are just the slowest and fastest test

INCOME_ORDER = [
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income",
]
INCOME_SHORT = {
    "Low income": "Low",
    "Lower middle income": "Lower-middle",
    "Upper middle income": "Upper-middle",
    "High income": "High",
}

# --------------------------------------------------------------------------
# Threshold configurations
# --------------------------------------------------------------------------
# Long form: one row per (use case, metric) with the requirement weight and
# threshold. Built from the iqb library's IQB_CONFIG so the default is exactly
# what the library, the web app and the Phase 2 report use.


def config_table(config: dict | None = None) -> pd.DataFrame:
    """Flatten an IQB config dict into a long table."""
    if config is None:
        from iqb import IQB_CONFIG

        config = IQB_CONFIG
    rows = []
    for uc, ucd in config["use cases"].items():
        for req, rd in ucd["network requirements"].items():
            rows.append(
                dict(
                    use_case=uc,
                    metric=REQ_TO_METRIC[req],
                    uc_weight=float(ucd["w"]),
                    req_weight=float(rd["w"]),
                    threshold=float(rd["threshold min"]),
                )
            )
    return pd.DataFrame(rows)


# Thresholds from the IQB framework table ("Designing the IQB framework",
# M-Lab Pulse slides, 2025-12-11, slide 10). Values in Mbit/s, ms, and loss as a
# fraction. "Other" (no specific high-quality value) keeps the minimum value.
# Video streaming's high-quality download is given as "50-100 Mb/s"; the upper
# end is used.
FRAMEWORK_MIN = {
    "web browsing": dict(download=10, upload=10, latency=100, loss=0.01),
    "video streaming": dict(download=25, upload=10, latency=100, loss=0.01),
    "video conferencing": dict(download=10, upload=25, latency=50, loss=0.005),
    "audio streaming": dict(download=10, upload=10, latency=100, loss=0.01),
    "online backup": dict(download=10, upload=25, latency=100, loss=0.01),
    "gaming": dict(download=10, upload=10, latency=100, loss=0.01),
}
FRAMEWORK_HIGH = {
    "web browsing": dict(download=100, upload=10, latency=50, loss=0.005),
    "video streaming": dict(download=100, upload=10, latency=50, loss=0.001),
    "video conferencing": dict(download=100, upload=100, latency=20, loss=0.001),
    "audio streaming": dict(download=50, upload=50, latency=50, loss=0.001),
    "online backup": dict(download=10, upload=200, latency=100, loss=0.001),
    "gaming": dict(download=100, upload=10, latency=50, loss=0.005),
}


def with_thresholds(tbl: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Copy of a config table with thresholds replaced (weights unchanged)."""
    out = tbl.copy()
    out["threshold"] = [thresholds[r.use_case][r.metric] for r in out.itertuples()]
    return out


def with_use_case_weights(tbl: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Copy of a config table with use-case weights replaced (missing -> 0)."""
    out = tbl.copy()
    out["uc_weight"] = out.use_case.map(lambda u: float(weights.get(u, 0.0)))
    return out


def scale_thresholds(tbl: pd.DataFrame, factor: float, metrics=METRICS) -> pd.DataFrame:
    """Make every threshold stricter (factor > 1) or more lenient (factor < 1).

    Throughput thresholds are multiplied by ``factor``; latency and loss
    thresholds are divided by it, so ``factor`` always means "stricter".
    """
    out = tbl.copy()
    for m in metrics:
        sel = out.metric == m
        out.loc[sel, "threshold"] = (
            out.loc[sel, "threshold"] * factor
            if HIGHER_IS_BETTER[m]
            else out.loc[sel, "threshold"] / factor
        )
    return out


def test_key(metric: str, threshold: float) -> str:
    op = ">" if HIGHER_IS_BETTER[metric] else "<"
    return f"{metric}{op}{threshold:g}"


def distinct_tests(*tables: pd.DataFrame) -> pd.DataFrame:
    """Distinct (metric, threshold) pairs used by one or more config tables."""
    t = pd.concat([x[["metric", "threshold"]] for x in tables]).drop_duplicates()
    t["key"] = [test_key(m, th) for m, th in zip(t.metric, t.threshold)]
    return t.sort_values(["metric", "threshold"]).reset_index(drop=True)


def effective_test_weights(tbl: pd.DataFrame) -> pd.Series:
    """Weight of each distinct test in the final IQB score (sums to 1)."""
    t = tbl.copy()
    t["req_share"] = t.req_weight / t.groupby("use_case").req_weight.transform("sum")
    uc_w = t.groupby("use_case").uc_weight.first()
    t["w"] = t.req_share * t.use_case.map(uc_w / uc_w.sum())
    t["key"] = [test_key(m, th) for m, th in zip(t.metric, t.threshold)]
    return t.groupby("key").w.sum().sort_values(ascending=False)


# --------------------------------------------------------------------------
# Loading the M-Lab percentile tables
# --------------------------------------------------------------------------

LEVEL_KEYS = {
    "country": ["country_code"],
    "country_city": [
        "country_code",
        "subdivision1_iso_code",
        "subdivision1_name",
        "city",
    ],
    "country_asn": ["country_code", "asn", "as_name"],
    "country_subdivision1": [
        "country_code",
        "subdivision1_iso_code",
        "subdivision1_name",
    ],
}


def month_dirs() -> dict[pd.Timestamp, Path]:
    """Map month start -> directory holding that month's tables."""
    out = {}
    for d in sorted(DATA_V1.glob("2*/2*")):
        out[pd.Timestamp(d.parent.name[:8])] = d
    return out


def _pcols(metric):
    return [f"{metric}_p{p}" for p in PCTS]


def load_level(level: str = "country", months=None) -> pd.DataFrame:
    """Load download + upload tables for a geographic level.

    Returns one row per (unit, month) with ``n_down`` (download-side tests,
    which also carry latency and loss), ``n_up`` (upload tests) and the nine
    reported percentiles of each metric. Units present in only one direction
    are dropped (an IQB score needs both).
    """
    keys = LEVEL_KEYS[level]
    md = month_dirs()
    if months is not None:
        months = {pd.Timestamp(m) for m in months}
        md = {m: d for m, d in md.items() if m in months}
    frames = []
    for m, d in md.items():
        dn = pd.read_parquet(
            d / f"downloads_by_{level}" / "data.parquet",
            columns=keys
            + ["sample_count"]
            + _pcols("download")
            + _pcols("latency")
            + _pcols("loss"),
        ).rename(columns={"sample_count": "n_down"})
        up = pd.read_parquet(
            d / f"uploads_by_{level}" / "data.parquet",
            columns=keys + ["sample_count"] + _pcols("upload"),
        ).rename(columns={"sample_count": "n_up"})
        if level == "country_city":  # a few city keys repeat; keep the larger row
            dn = dn.sort_values("n_down").drop_duplicates(keys, keep="last")
            up = up.sort_values("n_up").drop_duplicates(keys, keep="last")
        df = dn.merge(up, on=keys, how="inner")
        df.insert(len(keys), "month", m)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Namibia's ISO code "NA" must not be read as missing.
    out["country_code"] = out["country_code"].astype(str)
    return out


def read_table(name: str, **kw) -> pd.DataFrame:
    """Read a CSV written by an earlier notebook (keeps Namibia's code "NA")."""
    return pd.read_csv(TAB / f"{name}.csv", keep_default_na=False, na_values=[""], **kw)


def binding_tests(df: pd.DataFrame):
    """Tests that limit precision: the smaller of download-side and upload counts.

    Download, latency and loss pass rates rest on ``n_down``; upload pass rates
    on ``n_up`` (typically about 0.8 x n_down). The data-sufficiency rule must
    hold for every pass rate, so it applies to the smaller of the two.
    """
    return np.minimum(df["n_down"], df["n_up"])



# --------------------------------------------------------------------------
# Pass rates
# --------------------------------------------------------------------------
_EPS = 1e-6
LOSS_OFFSET = 1e-4  # packet loss is often exactly 0; log(loss + offset) keeps it finite


def _quality(metric: str, x):
    """Monotone map to a scale where higher is better and interpolation is most accurate.

    Log scale for every metric: in a held-out-percentile test (notebook 1a) it
    puts the hidden percentile on the correct side of a threshold more often
    than linear interpolation, for all four metrics.
    """
    x = np.asarray(x, dtype=float)
    if metric in ("download", "upload"):
        return np.log(np.maximum(x, _EPS))
    if metric == "latency":
        return -np.log(np.maximum(x, _EPS))
    return -np.log(np.maximum(x, 0.0) + LOSS_OFFSET)  # packet loss


def pass_rate(df: pd.DataFrame, metric: str, threshold: float) -> np.ndarray:
    """Share of tests whose ``metric`` beats ``threshold``, per row of ``df``.

    The quality quantile function is taken as piecewise linear (on the
    :func:`_quality` scale) between the nine reported percentiles, and flat
    beyond P1 and P99. The pass rate is the length of the percentile range on
    which it beats the threshold (strict inequality, as in IQBCalculator).
    """
    q = np.array([0.0, *[p / 100 for p in PCTS], 1.0])
    s = _quality(metric, df[_pcols(metric)].to_numpy())
    s = np.column_stack([s[:, :1], s, s[:, -1:]])  # flat tails
    st = _quality(metric, threshold)
    lo, hi = s[:, :-1], s[:, 1:]
    width = np.diff(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(
            hi > lo, np.clip((hi - st) / (hi - lo), 0.0, 1.0), (lo > st).astype(float)
        )
    return (frac * width).sum(axis=1)


def pass_rates(df: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Pass rate of every test in ``tests`` (from :func:`distinct_tests`)."""
    return pd.DataFrame(
        {r.key: pass_rate(df, r.metric, r.threshold) for r in tests.itertuples()},
        index=df.index,
    )



def pool_pass_rates(
    df: pd.DataFrame, pr: pd.DataFrame, by, tests: pd.DataFrame
) -> pd.DataFrame:
    """Pool pass rates over rows sharing ``by`` (exact test-weighted average).

    Upload tests are weighted by ``n_up``; all other metrics by ``n_down``.
    Also returns the pooled ``n_down``, ``n_up`` and number of rows pooled.
    """
    by = [by] if isinstance(by, str) else list(by)
    w = pd.DataFrame(index=df.index)
    for r in tests.itertuples():
        n = df["n_up"] if r.metric == "upload" else df["n_down"]
        w[r.key] = pr[r.key] * n
    w[by] = df[by]
    w["n_down"], w["n_up"] = df["n_down"], df["n_up"]
    g = w.groupby(by, sort=True)
    s = g.sum(numeric_only=True)
    out = pd.DataFrame(index=s.index)
    for r in tests.itertuples():
        out[r.key] = s[r.key] / (s["n_up"] if r.metric == "upload" else s["n_down"])
    out["n_down"], out["n_up"] = s["n_down"], s["n_up"]
    out["n_rows"] = g.size()
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _req_pass(pr: pd.DataFrame, tbl: pd.DataFrame, q: np.ndarray) -> np.ndarray:
    """Pass indicators, shape (units, requirements, percentiles)."""
    keys = [test_key(m, th) for m, th in zip(tbl.metric, tbl.threshold)]
    rates = pr[keys].to_numpy()[:, :, None]
    return (rates > 1.0 - q[None, None, :] + 1e-12).astype(float)


def use_case_scores(
    pr: pd.DataFrame, tbl: pd.DataFrame, q=PCTS
) -> dict[str, pd.DataFrame]:
    """Use-case scores at percentiles ``q``: {use case -> DataFrame(units x q)}."""
    qa = np.asarray(q, dtype=float) / 100
    passed = _req_pass(pr, tbl, qa)
    out = {}
    for uc in dict.fromkeys(tbl.use_case):
        idx = np.flatnonzero((tbl.use_case == uc).to_numpy())
        w = tbl.req_weight.to_numpy()[idx]
        sc = (passed[:, idx, :] * w[None, :, None]).sum(axis=1) / w.sum()
        out[uc] = pd.DataFrame(sc, index=pr.index, columns=list(q))
    return out


def iqb_scores(pr: pd.DataFrame, tbl: pd.DataFrame, q=PCTS) -> pd.DataFrame:
    """IQB score at percentiles ``q`` for every unit (rows) -> DataFrame(units x q)."""
    ucs = use_case_scores(pr, tbl, q)
    uc_w = tbl.groupby("use_case", sort=False).uc_weight.first()
    total = sum(ucs[uc] * w for uc, w in uc_w.items() if w > 0)
    return total / uc_w.sum()


def tipping_percentile(pr: pd.DataFrame, tbl: pd.DataFrame, use_case: str) -> pd.Series:
    """Lowest percentile (0-100) at which every requirement of a use case passes.

    A use case meets its thresholds at percentile q exactly when q exceeds this
    value. 0 means it passes at every percentile; 100 means at none.
    """
    g = tbl[tbl.use_case == use_case]
    keys = [test_key(m, th) for m, th in zip(g.metric, g.threshold)]
    return 100 * (1 - pr[keys].min(axis=1))


# --------------------------------------------------------------------------
# Confidence in a displayed score (used in 1b-sensitivity)
# --------------------------------------------------------------------------
_ENGINE = {}


def _score_engine():
    """Weights and tests of the default configuration, computed once."""
    if "default" not in _ENGINE:
        tbl = config_table()
        W = effective_test_weights(tbl)
        tests = distinct_tests(tbl).set_index("key").loc[W.index].reset_index()
        _ENGINE["default"] = (W, tests)
    return _ENGINE["default"]


def stable_seed(*parts) -> int:
    """Reproducible random seed from labels, so the same country gets the same draws everywhere."""
    import hashlib

    text = "|".join(
        f"{float(p):.3f}" if isinstance(p, (int, float, np.number)) else str(p)
        for p in parts
    )
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def score_uncertainty(
    row, n_tests, levels=(50,), key="", deff=None, B=2000, tol=None
) -> pd.DataFrame:
    """Displayed IQB score, confidence and 95 % range at each percentile level.

    ``row`` holds the pass rates of one unit (a country-year, a pooled window)
    plus ``n_down`` and ``n_up``; ``n_tests`` is the download-side test count to
    assume (uploads scale with the unit's own upload/download ratio). For each
    metric, the shares of tests between its thresholds are drawn from a
    Dirichlet distribution centred on the observed shares, with n / DEFF
    effective tests and a Jeffreys-type prior. Each draw gives a plausible true
    score; confidence is the share within ±tol of the displayed score.

    ``key`` labels the unit (e.g. "KE|2025"); together with n_tests and DEFF it
    seeds the draws, so results do not depend on which levels are requested or
    in which order units are processed.
    """
    tol = TOL if tol is None else tol
    W, observed, passed = _posterior_passes(row, n_tests, levels, key, deff, B)
    q = np.asarray(levels, dtype=float) / 100
    shown = (W.to_numpy()[:, None] * (observed[:, None] > 1 - q[None, :])).sum(0)
    scores = (passed * W.to_numpy()[None, :, None]).sum(1)

    closest = np.abs(observed[:, None] - (1 - q)[None, :]).argmin(0)
    return pd.DataFrame(
        {
            "score": shown,
            "conf": (np.abs(scores - shown[None, :]) <= tol + 1e-9).mean(0),
            "lo": np.percentile(scores, 2.5, axis=0),
            "hi": np.percentile(scores, 97.5, axis=0),
            "closest_test": W.index[closest],
            "closest_pass_rate": observed[closest],
        },
        index=pd.Index(list(levels), name="level"),
    )


def _posterior_passes(row, n_tests, levels, key, deff=None, B=2000):
    """Posterior draws of which of the ten tests pass, at each level: (weights, observed pass rates, draws).

    The draws are shared by :func:`score_uncertainty` and :func:`score_patterns`,
    so both describe exactly the same simulation.
    """
    deff = DEFF if deff is None else deff
    W, tests = _score_engine()
    q = np.asarray(levels, dtype=float) / 100
    observed = np.asarray(row[W.index], dtype=float)
    rng = np.random.default_rng(stable_seed(key, n_tests, deff, B))
    passed = np.zeros((B, len(W), len(q)), dtype=bool)
    ratio = float(row["n_up"]) / float(row["n_down"])
    for metric, g in tests.groupby("metric"):
        n_eff = max((n_tests * ratio if metric == "upload" else n_tests) / deff, 1.0)
        g = g.assign(rate=np.asarray(row[g.key], dtype=float)).sort_values(
            "rate", ascending=False
        )
        rates = np.minimum.accumulate(g.rate.to_numpy())
        shares = np.clip(
            -np.diff(np.concatenate([[1.0], rates, [0.0]])), 0, None
        )  # fail all ... pass all
        draws = rng.dirichlet(n_eff * shares + 0.5, size=B)
        pass_draws = draws[:, ::-1].cumsum(1)[:, ::-1][
            :, 1:
        ]  # pass rate of each threshold
        ok = pass_draws[:, :, None] > 1 - q[None, None, :]
        for j, k in enumerate(g.key):
            passed[:, W.index.get_loc(k), :] = ok[:, j, :]
    return W, observed, passed


def score_patterns(row, n_tests, levels=(50,), key="", deff=None, B=2000) -> dict:
    """The same simulation as :func:`score_uncertainty`, kept as pass/fail patterns of the ten tests.

    A pattern is a bit mask over the tests in ``test_order()`` (bit k = test k
    passes). For each level this returns the observed pattern and the distinct
    simulated patterns with their counts (out of ``B``). Because the IQB score is
    a weighted sum of passed tests, the score, confidence and range for *any*
    use-case weighting can be recomputed exactly from these few patterns.
    """
    W, observed, passed = _posterior_passes(row, n_tests, levels, key, deff, B)
    bits = 1 << np.arange(len(W), dtype=np.int64)
    q = np.asarray(levels, dtype=float) / 100
    obs_mask = ((observed[:, None] > 1 - q[None, :]) * bits[:, None]).sum(0)
    out = {}
    for j, lv in enumerate(levels):
        masks = (passed[:, :, j] * bits[None, :]).sum(1)
        vals, counts = np.unique(masks, return_counts=True)
        out[lv] = {
            "obs": int(obs_mask[j]),
            "dist": [[int(m), int(c)] for m, c in zip(vals, counts)],
        }
    return {"pass_rates": observed, "levels": out}


def test_order() -> list:
    """Test keys in the bit order used by :func:`score_patterns`."""
    return list(_score_engine()[0].index)


def display_decision(n_binding, conf) -> str:
    """How a score should be published: the minimum number of tests (1a-sample-size), then confidence."""
    if not np.isfinite(n_binding) or n_binding < FLOOR_TESTS:
        return "no score (too few tests)"
    if n_binding < MIN_TESTS:
        return "pool more data"
    return "show the score" if conf >= CONF else "show a range"


# --------------------------------------------------------------------------
# Small statistics helpers (scipy only)
# --------------------------------------------------------------------------


def spearman_ci(x, y, n_boot=2000, seed=0):
    """Spearman rho with a percentile bootstrap 95% CI (pairs resampled)."""
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    rx = np.argsort(np.argsort(x[idx], axis=1), axis=1).astype(float)
    ry = np.argsort(np.argsort(y[idx], axis=1), axis=1).astype(float)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    boots = (rx * ry).sum(1) / np.sqrt((rx**2).sum(1) * (ry**2).sum(1))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, lo, hi, len(x)


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y after removing (rank-)linear effects of controls."""
    from scipy.stats import pearsonr, rankdata

    d = pd.DataFrame({"x": x, "y": y})
    c = pd.DataFrame(controls)
    c.index = d.index
    d = pd.concat([d, c], axis=1).dropna()
    r = d.apply(rankdata)
    Z = np.column_stack([np.ones(len(r)), r.drop(columns=["x", "y"]).to_numpy()])
    res = {}
    for v in ("x", "y"):
        beta, *_ = np.linalg.lstsq(Z, r[v].to_numpy(), rcond=None)
        res[v] = r[v].to_numpy() - Z @ beta
    out = pearsonr(res["x"], res["y"])
    return out.statistic, out.pvalue, len(d)


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------


def norm_name(s: str) -> str:
    """Lower-case ASCII name with punctuation collapsed (for joining place names)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------
# Plot style (reference palette from the dataviz guidelines, light mode)
# --------------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
BLUE_RAMP = {
    100: "#cde2fb",
    150: "#b7d3f6",
    200: "#9ec5f4",
    250: "#86b6ef",
    300: "#6da7ec",
    350: "#5598e7",
    400: "#3987e5",
    450: "#2a78d6",
    500: "#256abf",
    550: "#1c5cab",
    600: "#184f95",
    650: "#104281",
    700: "#0d366b",
}
# Income groups are ordered, so they take an ordinal single-hue ramp.
INCOME_COLORS = dict(
    zip(INCOME_ORDER, [BLUE_RAMP[250], BLUE_RAMP[400], BLUE_RAMP[550], BLUE_RAMP[700]])
)
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")  # blue <-> red, gray midpoint


def set_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            # a list enables per-glyph fallback (arrows, Greek letters)
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def savefig(fig, name: str):
    fig.savefig(FIG / f"{name}.png")


def ecdf_xy(x):
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    return x, np.arange(1, len(x) + 1) / len(x)
