"""Generate the SIMULATED farm-level dataset used for factor analysis (Weeks 1 and 5).

IMPORTANT HONESTY NOTE
----------------------
Farm-level wheat data with soil, weather and management variables is not openly available for Punjab.
The task brief allows hypothetical data, so this script creates a synthetic dataset of 1,500 farms.
The yield equation below is MY OWN ASSUMPTION, built from general agronomic principles
(diminishing returns to nitrogen, water-limited yield, heat stress during grain fill, late-sowing
penalty). Only one coefficient is anchored to published research: roughly -6% wheat yield per +1 degC
of warming (Zhao et al., 2017, PNAS 114(35):9326-9331). The dataset is used to demonstrate the
analysis pipeline (EDA, modelling, validation). Any model trained on it will 'rediscover' these
assumptions, so its results are NOT new scientific evidence about real farms.

Farm districts are sampled in proportion to each district's real 2019-20 wheat area so the synthetic
farms are geographically consistent with the real PBS data.
"""
import numpy as np
import pandas as pd
from common import SIM, RAINFED, build_panel

rng = np.random.default_rng(20261009)   # fixed seed -> reproducible
N = 1500

panel = build_panel()
last = panel[(panel.year == "2019-20") & (panel.district != "Islamabad")]
w = last.area / last.area.sum()
district = rng.choice(last.district.values, size=N, p=w.values)
rainfed = np.array([d in RAINFED for d in district])

rain = np.where(rainfed, rng.normal(170, 55, N), rng.normal(85, 35, N)).clip(20, 400)      # Rabi-season rain (mm)
irrig = np.where(rainfed, rng.choice([0, 1], N, p=[.8, .2]), rng.choice([3, 4, 5, 6], N, p=[.2, .4, .3, .1]))
temp = rng.normal(20.5, 1.6, N).clip(15, 27)                                              # mean grain-fill temp (degC)
N_kg = rng.normal(105, 38, N).clip(0, 200)
P_kg = rng.normal(48, 18, N).clip(0, 100)
som = rng.normal(0.85, 0.28, N).clip(0.2, 2.0)                                            # soil organic matter (%)
ph = rng.normal(8.0, 0.35, N).clip(6.8, 9.0)
texture = rng.choice(["sandy", "loam", "clay"], N, p=[.2, .55, .25])
delay = rng.normal(9, 9, N).clip(-5, 45)                                                  # days after 15 Nov
improved = rng.random(N) < 0.6
weeding = rng.random(N) < 0.7
farm_ha = np.round(rng.lognormal(0.9, 0.7, N).clip(0.4, 40), 1)

eff_water = rain + irrig * 60.0                       # assumption: one irrigation ~ 60 mm
f_water = 1 - np.exp(-eff_water / 150.0)
f_temp = 1 - 0.06 * np.maximum(0, temp - 19) - 0.02 * np.maximum(0, 16 - temp)   # -6%/degC above 19 (anchor: Zhao 2017)
f_n = 1 - 0.30 * (1 - np.minimum(N_kg, 120) / 120) ** 2                              # diminishing returns, plateau at 120
f_p = 1 - 0.08 * (1 - np.minimum(P_kg, 60) / 60) ** 2
f_som = 1 + 0.12 * (som - 0.8)
f_ph = 1 - 0.10 * np.maximum(0, ph - 8.1)
f_sow = 1 - 0.007 * np.maximum(0, delay - 10)         # assumption: ~0.7% loss per day of delay after 25 Nov
f_var = np.where(improved, 1.08, 1.0)
f_weed = np.where(weeding, 1.05, 1.0)
f_tex = pd.Series(texture).map({"sandy": 0.95, "loam": 1.0, "clay": 1.01}).values
noise = rng.lognormal(0, 0.08, N)

POTENTIAL = 3900.0
yld = POTENTIAL * f_water * f_temp * f_n * f_p * f_som * f_ph * f_sow * f_var * f_weed * f_tex * noise

df = pd.DataFrame(dict(
    farm_id=[f"F{i:04d}" for i in range(1, N + 1)], district=district,
    zone=np.where(rainfed, "rainfed", "irrigated"), farm_size_ha=farm_ha,
    rainfall_mm=rain.round(0), irrigations=irrig, grain_fill_temp_c=temp.round(1),
    nitrogen_kg_ha=N_kg.round(0), phosphorus_kg_ha=P_kg.round(0), soil_om_pct=som.round(2), soil_ph=ph.round(2),
    soil_texture=texture, sowing_delay_days=delay.round(0), improved_variety=improved.astype(int),
    weed_control=weeding.astype(int), yield_kg_ha=yld.round(0)))
# realistic imperfection: ~2% missing in two soil columns (missing completely at random)
for col in ("soil_om_pct", "soil_ph"):
    df.loc[rng.random(N) < 0.02, col] = np.nan

df.to_csv(SIM / "simulated_farms.csv", index=False)
print(df.shape)
print(df.describe().T[["mean", "std", "min", "max"]].round(1))
