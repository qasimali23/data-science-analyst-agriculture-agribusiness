"""WEEK 2 - Strategic supply-chain analysis.

REAL inputs : PBS district production 2007-08..2019-20 (concentration, stability).
ASSUMED inputs (clearly labelled, to be replaced with field measurements): stage-wise post-harvest
loss rates and intervention effectiveness. They are calibrated against two published anchors:
 * FAO, State of Food and Agriculture 2019: ~14% of food is lost between harvest and retail globally;
   estimates reach 20-21% in Central and Southern Asia (all food commodities, not wheat alone).
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from common import *

R = {}
clean = build_panel()
tot = punjab_totals()

# ---------- real data: concentration of supply ----------
last3 = clean[clean.year.isin(YEARS[-3:])].groupby("district")["prod"].mean().sort_values(ascending=False)
share = last3 / last3.sum()
R["n_districts"] = int(len(share))
R["top10_share_pct"] = float(share.head(10).sum() * 100)
R["top5_share_pct"] = float(share.head(5).sum() * 100)
R["top10"] = [(d, round(float(v * 100), 1)) for d, v in share.head(10).items()]
R["hhi_district"] = float((share ** 2).sum() * 10000)
R["districts_for_50pct"] = int((share.cumsum() < 0.5).sum() + 1)
R["districts_for_80pct"] = int((share.cumsum() < 0.8).sum() + 1)
dshare = clean[clean.year.isin(YEARS[-3:])].groupby("division")["prod"].sum()
dshare = (dshare / dshare.sum() * 100).sort_values(ascending=False)
R["division_share_pct"] = {k: round(float(v), 1) for k, v in dshare.items()}
R["base_prod_kt_2019_20"] = float(tot["prod"].iloc[-1])
R["base_prod_kt_mean_3y"] = float(tot["prod"].iloc[-3:].mean())

# stability of supply: year-on-year drops in Punjab production
yoy = tot["prod"].diff().dropna()
R["worst_drop_kt"], R["worst_drop_year"] = float(yoy.min()), yoy.idxmin()
R["worst_drop_pct"] = float((tot["prod"].pct_change().min()) * 100)
R["n_years_with_drop"] = int((yoy < 0).sum())
R["n_year_pairs"] = int(len(yoy))
R["mean_drop_when_negative_kt"] = float(yoy[yoy < 0].mean())

cum = share.cumsum().values * 100
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(range(1, len(share) + 1), cum, color=BLUE, marker="o", ms=3.5, lw=2)
ax.set_xlabel("Number of districts, ranked from largest to smallest producer")
ax.set_ylabel("Cumulative share of Punjab wheat (%)")
for pct, nd in ((50, R["districts_for_50pct"]), (80, R["districts_for_80pct"])):
    ax.axhline(pct, color=INK2, lw=0.8, ls=":")
    ax.plot([nd, nd], [0, pct], color=ORANGE, lw=1.2, ls="--")
    ax.text(nd + 0.4, 4, f"{nd} districts = {pct}%", fontsize=8.5, color=ORANGE)
ax.set_ylim(0, 102)
ax.set_title("District concentration of wheat supply (average 2017-18 to 2019-20)")
savefig(fig, "w2_02_pareto.png")

# ---------- stylised loss model (ASSUMPTIONS) ----------
stages = ["Harvest &\nthreshing", "On-farm\nstorage", "Transport to\nmarket", "Mandi / procurement\nstorage", "Milling &\nprocessing", "Wholesale &\ndistribution"]
loss = np.array([0.030, 0.025, 0.010, 0.030, 0.020, 0.015])       # ASSUMED baseline stage loss rates
base = R["base_prod_kt_2019_20"]
remain = [base]
lost_kt = []
for l in loss:
    lost_kt.append(remain[-1] * l)
    remain.append(remain[-1] * (1 - l))
total_loss = 1 - np.prod(1 - loss)
R["stage_loss_rates_pct"] = [float(x * 100) for x in loss]
R["total_loss_pct"] = float(total_loss * 100)
R["total_loss_kt"] = float(base - remain[-1])
R["stage_loss_kt"] = [float(x) for x in lost_kt]
R["one_pp_kt"] = float(base * 0.01)

# interventions: share of THAT stage's loss removed (ASSUMED effectiveness)
interv = {
    "Harvest set-up & thresher calibration": (0, 0.30),
    "Hermetic / metal on-farm storage": (1, 0.50),
    "Route planning & covered transport": (2, 0.25),
    "Covered mandi storage + fumigation scheduling": (3, 0.40),
    "Mill intake quality control & moisture checks": (4, 0.15),
}
saved = {}
new_loss = loss.copy()
for name, (i, eff) in interv.items():
    l2 = loss.copy()
    l2[i] *= (1 - eff)
    saved[name] = float(base * (np.prod(1 - l2) - np.prod(1 - loss)))
    new_loss[i] *= (1 - eff)
comb = float(base * (np.prod(1 - new_loss) - np.prod(1 - loss)))
R["interventions_saved_kt"] = {k: round(v, 1) for k, v in saved.items()}
R["combined_saved_kt"] = comb
R["combined_new_total_loss_pct"] = float((1 - np.prod(1 - new_loss)) * 100)
R["combined_saved_pct_of_prod"] = comb / base * 100

# Monte-Carlo uncertainty: each stage loss ~ Triangular(0.5x, x, 1.8x) (ASSUMED bounds)
rng = np.random.default_rng(7)
S = 20000
draws = np.column_stack([rng.triangular(0.5 * l, l, 1.8 * l, S) for l in loss])
tot_l = 1 - np.prod(1 - draws, axis=1)
R["mc_total_loss_pct_p5_p50_p95"] = [float(np.percentile(tot_l, q) * 100) for q in (5, 50, 95)]
R["mc_rank_corr"] = {s.replace("\n", " "): round(float(stats.spearmanr(draws[:, i], tot_l)[0]), 2) for i, s in enumerate(stages)}

# ---------- figures ----------
# 1. flowchart of chain with data capture points
fig, ax = blank_ax(11, 4.6)
names = ["Inputs &\nplanting", "Production\n(field)", "Harvest &\non-farm storage", "Procurement\n& mandi", "Processing\n(flour mills)", "Distribution\n& retail"]
data = ["soil tests, seed\nsales, sowing date", "satellite NDVI, rain,\nirrigation logs, yield", "moisture, storage\nlosses, delay days", "arrivals, stock levels,\nqueue time, grade", "intake, extraction\nrate, downtime", "dispatch time, price,\nstock-outs, returns"]
kpi = ["input cost\nper ha", "yield gap\n(kg/ha)", "post-harvest\nloss %", "stock turnover\ndays", "mill capacity\nutilisation %", "order fill\nrate %"]
for i, (n, d, k) in enumerate(zip(names, data, kpi)):
    x = 0.25 + i * 1.78
    box(ax, x, 3.0, 1.5, 1.1, n, fc="#eef4fc", ec=BLUE, fs=9, bold=True)
    box(ax, x, 1.55, 1.5, 1.0, d, fc="#fdf1ec", ec=ORANGE, fs=7.5)
    box(ax, x, 0.2, 1.5, 0.9, k, fc="#e9f7f1", ec=AQUA, fs=8)
    if i < 5:
        arrow(ax, x + 1.5, 3.55, x + 1.78, 3.55)
    arrow(ax, x + 0.75, 3.0, x + 0.75, 2.55, color=INK2)
    arrow(ax, x + 0.75, 1.55, x + 0.75, 1.1, color=INK2)
ax.text(0.0, 4.4, "Stage", fontsize=8, color=INK2, rotation=90, va="top")
ax.text(0.0, 2.65, "Data to capture", fontsize=8, color=INK2, rotation=90, va="top")
ax.text(0.0, 1.1, "KPI", fontsize=8, color=INK2, rotation=90, va="top")
savefig(fig, "w2_01_chain_flow.png")

# 3. waterfall
fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(stages) + 2)
ax.bar(0, base, color=BLUE, width=0.6)
cumv = base
for i, lk in enumerate(lost_kt):
    ax.bar(i + 1, lk, bottom=cumv - lk, color=ORANGE, width=0.6)
    ax.text(i + 1, cumv + base * 0.01, f"-{lk:,.0f}", ha="center", fontsize=8, color=INK2)
    cumv -= lk
ax.bar(len(stages) + 1, cumv, color=AQUA, width=0.6)
ax.set_ylim(base * 0.8, base * 1.04)
ax.set_xticks(x)
ax.set_xticklabels(["Production\n2019-20"] + stages + ["Reaches\nconsumers"], fontsize=7.5)
ax.set_ylabel("'000 tonnes")
ax.text(0, base * 1.012, f"{base:,.0f}", ha="center", fontsize=8, color=INK2)
ax.text(len(stages) + 1, cumv + base * 0.01, f"{cumv:,.0f}", ha="center", fontsize=8, color=INK2)
ax.set_title("Illustrative loss waterfall, Punjab wheat (assumed stage loss rates, axis truncated)")
savefig(fig, "w2_03_waterfall.png")

# 4. tornado (rank correlation)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
items = sorted(R["mc_rank_corr"].items(), key=lambda kv: kv[1])
ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE, height=0.6)
ax.set_xlabel("Spearman rank correlation with total loss (20,000 simulations)")
ax.set_title("Which stage drives the uncertainty in total loss?")
savefig(fig, "w2_04_sensitivity.png")

# 5. interventions
fig, ax = plt.subplots(figsize=(8.5, 3.6))
names_i = list(saved.keys()) + ["All combined"]
vals = list(saved.values()) + [comb]
cols = [BLUE] * len(saved) + [AQUA]
ax.barh(names_i[::-1], vals[::-1], color=cols[::-1], height=0.6)
for i, v in enumerate(vals[::-1]):
    ax.text(v + max(vals) * 0.01, i, f"{v:,.0f}", va="center", fontsize=8, color=INK2)
ax.set_xlabel("Wheat saved, '000 tonnes per season (on the 2019-20 base)")
ax.set_title("Estimated benefit of each data-guided intervention")
ax.set_xlim(0, max(vals) * 1.15)
savefig(fig, "w2_05_interventions.png")

save_result("week2", R)
print({k: R[k] for k in ["top10_share_pct", "hhi_district", "districts_for_50pct", "total_loss_pct", "combined_saved_kt",
                          "combined_new_total_loss_pct", "mc_total_loss_pct_p5_p50_p95", "worst_drop_kt", "worst_drop_year"]})
