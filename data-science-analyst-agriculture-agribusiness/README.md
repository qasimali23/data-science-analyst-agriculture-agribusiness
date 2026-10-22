# Punjab Wheat Analytics

Six-week data science project for the **Junior Data Science Analyst - Agriculture & Agribusiness** internship (09 Oct to 20 Nov 2026).
It studies wheat in Punjab, Pakistan: yield drivers, supply chain, market research, risk, prediction and an integrated decision-support view.

## Data (read this first)

| File | Type | Notes |
|---|---|---|
| `data/raw/yield_kg_per_ha.csv`, `production_1000t.csv`, `area_1000ha.csv` | **Real** | Pakistan Bureau of Statistics, *Social Indicators of Punjab*, district-wise wheat data 2007-08 to 2019-20 (republished at opendata.com.pk). The three tables agree with each other (yield = production / area). |
| `data/simulated/simulated_farms.csv` | **Simulated** | 1,500 synthetic farms. Open farm-level soil/weather/management data do not exist for Pakistan, so I generated them (`src/gen_simulated_farms.py`). The yield equation is my own assumption; only the heat effect (about -6% per +1 degC) is anchored to Zhao et al. 2017 (PNAS). Results on this file test the *method*, not real farms. |

Assumptions (stage loss rates, initiative effectiveness, rain-fed zone = Rawalpindi Division + Islamabad, 20% yield-gap closure, risk-matrix scores) are labelled in the code and the reports.

**Data-quality finding:** the published 2009-10 area values for Kasur (64) and Lahore (210) appear swapped, giving impossible yields of 8,906 and 781 kg/ha. `src/common.py` corrects this and the Week 1 report explains it.

## Layout

```
data/raw/                 real PBS tables
data/simulated/           simulated farm data
src/common.py             loading, cleaning, plotting helpers
src/gen_simulated_farms.py
src/w1_eda.py             Week 1  - exploratory data analysis
src/w2_supply_chain.py    Week 2  - supply chain concentration, loss model, Monte Carlo
src/w3_market.py          Week 3  - growth, k-means segments, survey sample design
src/w4_risk.py            Week 4  - shocks, zone risk, trend tests, scenario replay, risk register
src/w5_model.py           Week 5  - farm-level model (simulated) and district forecast (real)
src/w6_integration.py     Week 6  - district action matrix and portfolio view
outputs/figures/          all charts
outputs/tables/           CSV tables
outputs/results.json      every number quoted in the reports
reports/                  the six weekly Word reports
build/                    Node scripts that write the Word reports from results.json
```

## Run

```bash
pip install -r requirements.txt
./run_all.sh            # regenerates data, figures, tables and results.json
cd build && npm install && for i in 1 2 3 4 5 6; do node report$i.js; done   # optional: rebuild the reports
```

## Main results

- Punjab wheat yield rose about 33 kg/ha per year (2007-08 to 2019-20); area was flat, so yield drives output.
- Irrigated districts average about 2,863 kg/ha, rain-fed areas about 1,477 kg/ha, and rain-fed areas swing about 2.8x as much.
- Under assumed loss rates the chain loses about 12.3% of the crop; five initiatives could save about 691 thousand tonnes a season.
- A one-in-twenty shortfall is about 1,071 thousand tonnes in the replay (more under a cautious estimate).
- On real district data a ridge model beats naive forecasts by about 6-9% (RMSE 241 vs 265 kg/ha).

## Limits

Real data end in 2019-20 and contain no weather or price variables. The farm-level model uses simulated data. See each report's limitations section.

## Licence / source credit

Code: MIT-style, free to reuse with credit. Data: Pakistan Bureau of Statistics; cite the original source.
