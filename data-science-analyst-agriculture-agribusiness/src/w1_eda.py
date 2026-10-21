"""WEEK 1 - Exploratory data analysis of crop-yield factors.

Part A: REAL data  - PBS district-wise wheat area/production/yield, Punjab, 2007-08 to 2019-20.
Part B: SIMULATED data - 1,500 synthetic farms with soil, weather and management variables.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from common import *

R = {}
panel = build_panel(fix_2009_swap=False)      # as published
clean = build_panel(fix_2009_swap=True)       # corrected
tot = punjab_totals()

# ---------------- A1. Punjab-level trend ----------------
t = np.arange(len(YEARS))
slope, intercept, r, p, se = stats.linregress(t, tot["yld"].values)
R["punjab_yield_first"], R["punjab_yield_last"] = float(tot.yld.iloc[0]), float(tot.yld.iloc[-1])
R["punjab_yield_mean"], R["punjab_yield_sd"] = float(tot.yld.mean()), float(tot.yld.std(ddof=1))
R["punjab_yield_cv_pct"] = float(tot.yld.std(ddof=1) / tot.yld.mean() * 100)
R["punjab_yield_slope"], R["punjab_yield_slope_p"], R["punjab_yield_r2"] = float(slope), float(p), float(r ** 2)
R["punjab_yield_best_year"], R["punjab_yield_best"] = tot.yld.idxmax(), float(tot.yld.max())
R["punjab_yield_worst_year"], R["punjab_yield_worst"] = tot.yld.idxmin(), float(tot.yld.min())
R["punjab_prod_mean"], R["punjab_prod_min"], R["punjab_prod_max"] = float(tot["prod"].mean()), float(tot["prod"].min()), float(tot["prod"].max())
R["punjab_prod_min_year"], R["punjab_prod_max_year"] = tot["prod"].idxmin(), tot["prod"].idxmax()
R["punjab_area_mean"], R["punjab_area_min"], R["punjab_area_max"] = float(tot.area.mean()), float(tot.area.min()), float(tot.area.max())
R["punjab_area_cv_pct"] = float(tot.area.std(ddof=1) / tot.area.mean() * 100)
R["punjab_prod_cv_pct"] = float(tot["prod"].std(ddof=1) / tot["prod"].mean() * 100)
# what drives year-to-year production change: area or yield?  (log-difference decomposition)
dl = np.log(tot).diff().dropna()
R["corr_dprod_dyield"] = float(dl["prod"].corr(dl["yld"]))
R["corr_dprod_darea"] = float(dl["prod"].corr(dl["area"]))
R["corr_prod_area_levels"] = float(tot["prod"].corr(tot["area"]))
R["corr_prod_yield_levels"] = float(tot["prod"].corr(tot["yld"]))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for ax, col, ttl, col_c, unit in zip(axes, ["area", "prod", "yld"],
                                     ["Area sown", "Production", "Yield"], [BLUE, ORANGE, AQUA],
                                     ["'000 ha", "'000 tonnes", "kg/ha"]):
    ax.plot(range(len(YEARS)), tot[col].values, color=col_c, lw=2, marker="o", ms=4)
    ax.set_title(ttl)
    ax.set_ylabel(unit)
    ax.set_xticks(range(0, len(YEARS), 2))
    ax.set_xticklabels([YEARS[i][2:] for i in range(0, len(YEARS), 2)])
axes[2].plot(range(len(YEARS)), intercept + slope * t, color=INK2, lw=1, ls="--")
axes[2].text(0.3, tot.yld.max() * 0.985, f"trend {slope:+.1f} kg/ha per year", fontsize=8, color=INK2)
fig.suptitle("Punjab wheat, 2007-08 to 2019-20 (PBS)", x=0.01, ha="left", fontsize=10, color=INK2)
savefig(fig, "w1_01_punjab_trends.png")

# ---------------- A2. Data quality ----------------
y, p_, a = load_wide()
R["n_districts"] = int(len(clean.district.unique()))
R["n_cells_total"] = int(y.drop(index="PUNJAB").size)
R["n_missing_yield_cells"] = int(y.drop(index="PUNJAB").isna().sum().sum())
R["missing_cells_detail"] = [f"{d} {yr}" for d in y.index for yr in YEARS if pd.isna(y.loc[d, yr]) and d != "PUNJAB"]
calc = p_ / a * 1000
rel = ((y / calc) - 1).abs().stack()
R["cells_checked_consistency"] = int(rel.count())
R["cells_inconsistent_gt3pct"] = int((rel > 0.03).sum())
R["max_inconsistency_pct"] = float(rel.max() * 100)

# anomaly detection: robust z-score of each district-year yield against that district's own history
def robust_z(g):
    med, mad = g.median(), (g - g.median()).abs().median()
    return (g - med) / (1.4826 * mad)
panel["rz"] = panel.groupby("district")["yld"].transform(robust_z)
anom = panel[panel.rz.abs() > 3.5].sort_values("rz")
R["anomalies_published"] = [dict(district=r.district, year=r.year, yld=float(r.yld), rz=round(float(r.rz), 1))
                            for r in anom.itertuples()]
R["kasur_2009_pub_yield"] = float(panel[(panel.district == "Kasur") & (panel.year == "2009-10")].yld.iloc[0])
R["lahore_2009_pub_yield"] = float(panel[(panel.district == "Lahore") & (panel.year == "2009-10")].yld.iloc[0])
R["kasur_2009_fixed_yield"] = float(clean[(clean.district == "Kasur") & (clean.year == "2009-10")].yld.iloc[0])
R["lahore_2009_fixed_yield"] = float(clean[(clean.district == "Lahore") & (clean.year == "2009-10")].yld.iloc[0])
R["kasur_area_other_years_median"] = float(a.loc["Kasur"].drop("2009-10").median())
R["lahore_area_other_years_median"] = float(a.loc["Lahore"].drop("2009-10").median())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, d, c in zip(axes, ["Kasur", "Lahore"], [BLUE, ORANGE]):
    ax.plot(range(len(YEARS)), a.loc[d].values, color=c, marker="o", ms=4, lw=1.6)
    ax.scatter([2], [a.loc[d, "2009-10"]], s=70, facecolor="none", edgecolor=INK, lw=1.6, zorder=5)
    ax.set_title(f"{d}: published area sown")
    ax.set_ylabel("'000 hectares")
    ax.set_xticks(range(0, len(YEARS), 2))
    ax.set_xticklabels([YEARS[i][2:] for i in range(0, len(YEARS), 2)])
    ax.annotate("2009-10 value looks swapped", (2, a.loc[d, "2009-10"]), xytext=(3.2, a.loc[d].max() * 0.8),
                fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
fig.suptitle("Data-quality flag: Kasur and Lahore area in 2009-10", x=0.01, ha="left", fontsize=10, color=INK2)
savefig(fig, "w1_02_area_swap.png")

# ---------------- A3. District summaries ----------------
g = clean.groupby("district")
ds = pd.DataFrame({
    "division": g.division.first(), "rainfed": g.rainfed.first(),
    "mean_yield": g.yld.mean(), "sd_yield": g.yld.std(ddof=1), "min_yield": g.yld.min(), "max_yield": g.yld.max(),
    "mean_area_kha": g.area.mean(), "mean_prod_kt": g["prod"].mean()})
ds["cv_pct"] = ds.sd_yield / ds.mean_yield * 100
ds = ds.sort_values("mean_yield", ascending=False)
ds.round(1).to_csv(TAB / "w1_district_summary.csv")
R["top5_districts"] = [(d, round(float(v))) for d, v in ds.mean_yield.head(5).items()]
R["bottom5_districts"] = [(d, round(float(v))) for d, v in ds.mean_yield.tail(5).items()]
R["district_yield_range"] = [float(ds.mean_yield.min()), float(ds.mean_yield.max())]
R["district_cv_median"] = float(ds.cv_pct.median())
R["highest_cv"] = [(d, round(float(v), 1)) for d, v in ds.cv_pct.sort_values(ascending=False).head(5).items()]
R["lowest_cv"] = [(d, round(float(v), 1)) for d, v in ds.cv_pct.sort_values().head(5).items()]

rain = ds[ds.rainfed].mean_yield
irr = ds[~ds.rainfed].mean_yield
R["rainfed_mean_yield"], R["irrigated_mean_yield"] = float(rain.mean()), float(irr.mean())
R["rainfed_n"], R["irrigated_n"] = int(len(rain)), int(len(irr))
R["rainfed_vs_irrigated_gap_pct"] = float((1 - rain.mean() / irr.mean()) * 100)
u = stats.mannwhitneyu(rain, irr, alternative="two-sided")
R["mannwhitney_p"] = float(u.pvalue)
R["rainfed_cv_mean"], R["irrigated_cv_mean"] = float(ds[ds.rainfed].cv_pct.mean()), float(ds[~ds.rainfed].cv_pct.mean())

fig, ax = plt.subplots(figsize=(7.2, 8.2))
dd = ds.sort_values("mean_yield")
cols = [ORANGE if r else BLUE for r in dd.rainfed]
ax.hlines(range(len(dd)), dd.min_yield, dd.max_yield, color=GRID, lw=5, zorder=1)
ax.scatter(dd.mean_yield, range(len(dd)), c=cols, s=28, zorder=3)
ax.set_yticks(range(len(dd)))
ax.set_yticklabels(dd.index, fontsize=7.5)
ax.set_xlabel("Wheat yield, kg/ha (dot = 13-year mean, bar = min to max)")
ax.scatter([], [], c=BLUE, label="Irrigated (canal / tubewell)")
ax.scatter([], [], c=ORANGE, label="Rain-fed (Rawalpindi Division, Islamabad)")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("District yield level and range, 2007-08 to 2019-20")
savefig(fig, "w1_03_district_yield.png")

# ---------------- A4. Division heatmap + synchrony ----------------
dv = clean.groupby(["division", "year"]).apply(lambda x: x["prod"].sum() / x.area.sum() * 1000, include_groups=False).unstack()
dv = dv[YEARS]
dv.round(0).to_csv(TAB / "w1_division_yield.csv")
fig, ax = plt.subplots(figsize=(9, 3.8))
im = ax.imshow(dv.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(YEARS)))
ax.set_xticklabels([y_[2:] for y_ in YEARS], fontsize=8)
ax.set_yticks(range(len(dv)))
ax.set_yticklabels(dv.index, fontsize=8)
ax.grid(False)
for i in range(dv.shape[0]):
    for j in range(dv.shape[1]):
        v = dv.values[i, j]
        ax.text(j, i, f"{v / 1000:.1f}", ha="center", va="center", fontsize=7, color="white" if v > 2800 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.025)
cb.set_label("kg/ha")
ax.set_title("Division-level yield (tonnes/ha) by season")
savefig(fig, "w1_04_division_heatmap.png")

irr_div = dv.drop(index="Rawalpindi")
ch = irr_div.pct_change(axis=1).iloc[:, 1:]
cm = ch.T.corr().values
R["synchrony_mean_corr_irrigated_divisions"] = float(cm[np.triu_indices_from(cm, 1)].mean())
R["rawalpindi_div_yield_cv_pct"] = float(dv.loc["Rawalpindi"].std(ddof=1) / dv.loc["Rawalpindi"].mean() * 100)
R["irrigated_div_yield_cv_pct_mean"] = float((irr_div.std(axis=1, ddof=1) / irr_div.mean(axis=1) * 100).mean())
R["rawalpindi_2009_10_yield"] = float(dv.loc["Rawalpindi", "2009-10"])
R["rawalpindi_mean_other_years"] = float(dv.loc["Rawalpindi"].drop("2009-10").mean())
R["rawalpindi_2009_drop_pct"] = float((1 - R["rawalpindi_2009_10_yield"] / R["rawalpindi_mean_other_years"]) * 100)

# area vs yield relationship across districts
R["corr_area_yield_districts"] = float(ds.mean_area_kha.corr(ds.mean_yield, method="spearman"))

# ---------------- B. Simulated farm data ----------------
f = pd.read_csv(SIM / "simulated_farms.csv")
R["sim_n"] = int(len(f))
R["sim_missing_pct"] = {c: round(float(f[c].isna().mean() * 100), 1) for c in ("soil_om_pct", "soil_ph")}
R["sim_yield_mean"], R["sim_yield_sd"] = float(f.yield_kg_ha.mean()), float(f.yield_kg_ha.std())
R["sim_yield_q"] = [float(f.yield_kg_ha.quantile(q)) for q in (.05, .25, .5, .75, .95)]
R["sim_zone_yield"] = {z: round(float(v)) for z, v in f.groupby("zone").yield_kg_ha.mean().items()}
num = ["rainfall_mm", "irrigations", "grain_fill_temp_c", "nitrogen_kg_ha", "phosphorus_kg_ha", "soil_om_pct", "soil_ph",
       "sowing_delay_days", "improved_variety", "weed_control", "farm_size_ha", "yield_kg_ha"]
corr_p = f[num].corr()
corr_s = f[num].corr(method="spearman")
R["sim_corr_with_yield_pearson"] = {k: round(float(v), 2) for k, v in corr_p["yield_kg_ha"].drop("yield_kg_ha").items()}
R["sim_corr_with_yield_spearman"] = {k: round(float(v), 2) for k, v in corr_s["yield_kg_ha"].drop("yield_kg_ha").items()}

fig, ax = plt.subplots(figsize=(7.6, 6))
im = ax.imshow(corr_p.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num)))
ax.set_yticks(range(len(num)))
lab = [n.replace("_", " ") for n in num]
ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=8)
ax.set_yticklabels(lab, fontsize=8)
ax.grid(False)
for i in range(len(num)):
    for j in range(len(num)):
        ax.text(j, i, f"{corr_p.values[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color=INK)
fig.colorbar(im, ax=ax, fraction=0.04)
ax.set_title("Correlation matrix, simulated farms (Pearson)")
savefig(fig, "w1_05_sim_corr.png")

fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
zc = {"irrigated": BLUE, "rainfed": ORANGE}
for ax, col, xl in zip(axes.ravel(), ["nitrogen_kg_ha", "grain_fill_temp_c", "rainfall_mm", "sowing_delay_days"],
                       ["Nitrogen applied (kg/ha)", "Mean grain-fill temperature (degC)", "Rabi rainfall (mm)", "Sowing delay after 15 Nov (days)"]):
    for z, c in zc.items():
        s = f[f.zone == z]
        ax.scatter(s[col], s.yield_kg_ha, s=7, alpha=0.35, color=c, label=z, linewidths=0)
    ax.set_xlabel(xl)
    ax.set_ylabel("Yield (kg/ha)")
axes[0, 0].legend(markerscale=2.5, fontsize=8)
fig.suptitle("Yield against four candidate drivers (simulated farms)", x=0.01, ha="left", fontsize=10, color=INK2)
savefig(fig, "w1_06_sim_scatter.png")

fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
for ax, col, ttl in zip(axes, ["irrigations", "improved_variety", "soil_texture", "weed_control"],
                        ["Irrigations", "Improved variety (1=yes)", "Soil texture", "Weed control (1=yes)"]):
    order = sorted(f[col].unique(), key=str)
    data = [f[f[col] == o].yield_kg_ha.values for o in order]
    bp = ax.boxplot(data, tick_labels=[str(o) for o in order], patch_artist=True, widths=0.6, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor="#dbe8f8", edgecolor=BLUE)
    for m in bp["medians"]:
        m.set(color=INK, lw=1.5)
    ax.set_title(ttl, fontsize=9)
axes[0].set_ylabel("Yield (kg/ha)")
fig.suptitle("Yield by management and soil categories (simulated farms)", x=0.01, ha="left", fontsize=10, color=INK2)
savefig(fig, "w1_07_sim_box.png")

# group effects (differences in medians) for the report text
R["sim_improved_gain_pct"] = float((f[f.improved_variety == 1].yield_kg_ha.mean() / f[f.improved_variety == 0].yield_kg_ha.mean() - 1) * 100)
R["sim_weed_gain_pct"] = float((f[f.weed_control == 1].yield_kg_ha.mean() / f[f.weed_control == 0].yield_kg_ha.mean() - 1) * 100)
R["sim_texture_means"] = {k: round(float(v)) for k, v in f.groupby("soil_texture").yield_kg_ha.mean().items()}
R["sim_irrig_means"] = {int(k): round(float(v)) for k, v in f.groupby("irrigations").yield_kg_ha.mean().items()}
hot = f[f.grain_fill_temp_c >= 22].yield_kg_ha.mean()
cool = f[f.grain_fill_temp_c <= 19.5].yield_kg_ha.mean()
R["sim_hot_vs_cool_gap_pct"] = float((1 - hot / cool) * 100)
R["sim_n_hot"], R["sim_n_cool"] = int((f.grain_fill_temp_c >= 22).sum()), int((f.grain_fill_temp_c <= 19.5).sum())
lowN, highN = f[f.nitrogen_kg_ha < 60].yield_kg_ha.mean(), f[f.nitrogen_kg_ha >= 120].yield_kg_ha.mean()
R["sim_lowN_vs_highN_gap_pct"] = float((1 - lowN / highN) * 100)
R["sim_lateN"] = int((f.sowing_delay_days > 25).sum())

# standardized multiple regression (EDA-level driver ranking, on complete cases)
cc = f.dropna().copy()
X = cc[["rainfall_mm", "irrigations", "grain_fill_temp_c", "nitrogen_kg_ha", "phosphorus_kg_ha", "soil_om_pct", "soil_ph",
        "sowing_delay_days", "improved_variety", "weed_control"]].astype(float)
Xs = (X - X.mean()) / X.std()
ys = (cc.yield_kg_ha - cc.yield_kg_ha.mean()) / cc.yield_kg_ha.std()
A = np.column_stack([np.ones(len(Xs)), Xs.values])
beta, *_ = np.linalg.lstsq(A, ys.values, rcond=None)
res = ys.values - A @ beta
R["sim_ols_r2"] = float(1 - res.var() / ys.var())
R["sim_std_betas"] = {c: round(float(b), 2) for c, b in zip(Xs.columns, beta[1:])}

save_result("week1", R)
print("week1 done")
print({k: R[k] for k in ["punjab_yield_slope", "punjab_yield_slope_p", "corr_dprod_dyield", "corr_dprod_darea",
                          "rainfed_vs_irrigated_gap_pct", "anomalies_published", "cells_inconsistent_gt3pct",
                          "sim_std_betas", "sim_ols_r2", "synchrony_mean_corr_irrigated_divisions"]})
