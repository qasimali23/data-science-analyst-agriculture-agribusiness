"""Shared helpers: data loading, cleaning, plotting style, small drawing tools.

Project: Punjab Wheat Analytics (agriculture & agribusiness internship project).
All 'real' data comes from the Pakistan Bureau of Statistics (PBS), Social Indicators
of Punjab, as published on opendata.com.pk (district-wise wheat area, production and
yield, 2007-08 to 2019-20).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SIM = ROOT / "data" / "simulated"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
RESULTS = ROOT / "outputs" / "results.json"
for d in (FIG, TAB, SIM):
    d.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (slots 1-3 are safe for all-pairs use)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, GRID = "#1b1b1a", "#52514e", "#e4e3df"

YEARS = ["2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14",
         "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20"]

# Division membership follows the row order of the PBS table.
DIVISIONS = {
    "Bahawalpur": ["Bahawalnagar", "Bahawalpur", "Rahim Yar Khan"],
    "D.G.Khan": ["D.G.Khan", "Layyah", "Muzaffargarh", "Rajanpur"],
    "Faisalabad": ["Faisalabad", "Chiniot", "Jhang", "T.T.Singh"],
    "Gujranwala": ["Gujranwala", "Gujrat", "Hafizabad", "Mandi Baha-ud-Din", "Narowal", "Sialkot"],
    "Lahore": ["Kasur", "Lahore", "Nankana Sahib", "Sheikhupura"],
    "Multan": ["Khanewal", "Lodhran", "Multan", "Vehari"],
    "Rawalpindi": ["Attock", "Chakwal", "Jhelum", "Rawalpindi"],
    "Sahiwal": ["Sahiwal", "Okara", "Pakpattan"],
    "Sargodha": ["Bhakkar", "Khushab", "Mianwali", "Sargodha"],
}
DIST2DIV = {d: v for v, ds in DIVISIONS.items() for d in ds}
DIST2DIV["Islamabad"] = "Islamabad (ICT)"
# ASSUMPTION (documented in the reports): districts of Rawalpindi Division (Potohar plateau)
# plus Islamabad are treated as rain-fed ('barani') wheat zones; all others as canal/tubewell irrigated.
RAINFED = set(DIVISIONS["Rawalpindi"]) | {"Islamabad"}


def _load(name):
    return pd.read_csv(RAW / name, thousands=",", na_values=["**", "-"]).set_index("district")


def load_wide():
    """Return (yield, production, area) wide tables exactly as published (district rows only
    plus PUNJAB total). Division rows are dropped (we rebuild them ourselves)."""
    y, p, a = (_load(f) for f in ("yield_kg_per_ha.csv", "production_1000t.csv", "area_1000ha.csv"))
    keep = [i for i in y.index if "Div" not in i]
    return y.loc[keep], p.loc[keep], a.loc[keep]


def build_panel(fix_2009_swap=True):
    """Long district-year panel with columns: district, division, rainfed, year, area, prod, yield.

    Data-quality fix (documented in Week 1): in 2009-10 the published AREA values of Kasur (64) and
    Lahore (210) appear swapped relative to every other year (Kasur ~180, Lahore ~60). Published
    yields for those two cells (8,906 and 781 kg/ha) are therefore impossible. When fix_2009_swap is
    True we swap the two area values back and recompute yield = production / area.
    """
    y, p, a = load_wide()
    a = a.copy()
    if fix_2009_swap:
        a.loc["Kasur", "2009-10"], a.loc["Lahore", "2009-10"] = a.loc["Lahore", "2009-10"], a.loc["Kasur", "2009-10"]
    rows = []
    for d in y.index:
        if d == "PUNJAB":
            continue
        for yr in YEARS:
            ar, pr, yl = a.loc[d, yr], p.loc[d, yr], y.loc[d, yr]
            if fix_2009_swap and d in ("Kasur", "Lahore") and yr == "2009-10":
                yl = pr / ar * 1000
            rows.append(dict(district=d, division=DIST2DIV[d], rainfed=d in RAINFED, year=yr,
                             area=ar, prod=pr, yld=yl))
    df = pd.DataFrame(rows)
    df["t"] = df["year"].map({v: i for i, v in enumerate(YEARS)})
    return df


def punjab_totals():
    y, p, a = load_wide()
    return pd.DataFrame({"area": a.loc["PUNJAB"], "prod": p.loc["PUNJAB"], "yld": y.loc["PUNJAB"]})


def save_result(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def load_results():
    return json.loads(RESULTS.read_text())


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "font.size": 10, "legend.frameon": False, "axes.axisbelow": True,
    })


style()


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


# ---------- simple flowchart helpers ----------
def box(ax, x, y, w, h, text, fc="#eef4fc", ec=BLUE, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=INK2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def blank_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.grid(False)
    return fig, ax
