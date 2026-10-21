# Punjab Wheat Analytics

Data science internship project (Junior Data Science Analyst, Agriculture & Agribusiness). The project studies wheat in Punjab, Pakistan. More analysis is added each week.

## Week 1: Exploratory data analysis of crop yield factors

**Real data** (`data/raw/`): Pakistan Bureau of Statistics, *Social Indicators of Punjab* - district-wise wheat area, production and yield per hectare, 2007-08 to 2019-20 (republished at opendata.com.pk).

**Simulated data** (`data/simulated/simulated_farms.csv`): 1,500 synthetic farms with soil, weather and management variables. Open farm-level data do not exist for Pakistan, so I generated them with `src/gen_simulated_farms.py`. The yield equation is my own assumption; only the heat effect (about -6% per +1 degC) is anchored to Zhao et al. 2017 (PNAS). These data test the method and prove nothing about real farms.

**Data-quality finding:** the published 2009-10 area values for Kasur (64) and Lahore (210) appear swapped, giving impossible yields of 8,906 and 781 kg/ha. `src/common.py` corrects this.

### Run
```bash
pip install -r requirements.txt
cd src
python gen_simulated_farms.py
python w1_eda.py
```
Outputs: `outputs/figures/` (charts), `outputs/tables/` (CSV tables), `outputs/results.json` (numbers quoted in the report). The report is in `reports/`.

### Main Week 1 results
- Punjab wheat yield rose about 33 kg/ha per year; area was flat, so yield drives production.
- Irrigated districts average about 2,863 kg/ha; rain-fed areas about 1,477 kg/ha and far less stable.
- Divisions share good and bad seasons (average correlation 0.59), which points to weather as the main driver.
