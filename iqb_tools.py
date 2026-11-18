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
