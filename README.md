# Transport-Sector ML Portfolio — Piyush Jaangid

Three end-to-end machine learning case studies grounded in real transport and clean-mobility problems. Each notebook follows the full 13-step ML pipeline (Problem → Data → EDA → Cleaning → Features → Models → Train → Evaluate → Improve → Test → Deploy → Monitor) in a Khanna-Justo worked-example style — every input value cited, every formula written out, every step justified.

## What's inside

| # | Type | Notebook | Algorithms compared |
|---|---|---|---|
| 1 | **Regression** | `01_regression_trip_duration.ipynb` | Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| 2 | **Classification** | `02_classification_ebus_tco.ipynb` | Logistic, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting, Naive Bayes |
| 3 | **Unsupervised** | `03_unsupervised_taz_clustering.ipynb` | K-Means, Hierarchical (Ward), DBSCAN, Gaussian Mixture |

A unified Streamlit dashboard (`dashboard/app.py`) lets recruiters interact with all three without setting up Python.

## Case study summaries

### 1. Predicting Ride-Hail Trip Duration (Regression)

A ride-hailing operator needs accurate ETAs at booking time to reduce abandonment. We build a regression model on hourly aggregated trip data structured like the NYC TLC Trip Records, comparing seven algorithms on MAE, RMSE, and R². Tuned XGBoost achieves test MAE ~13 seconds and R² ~0.96, beating a city-mean baseline by roughly 70%. Includes a manual OLS sanity check, multicollinearity pruning, log-transform of the skewed target, and feature importance interpretation.

**Real data source pointer:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

### 2. Will an E-Bus Beat Diesel on TCO? (Classification)

A State Transport Undertaking procuring 1,000 buses needs to decide route-by-route between electric and diesel. We build a Total Cost of Ownership model parameterised from WRI India and ICCT publications (capex, FAME-II subsidies, energy and maintenance costs, mid-life battery replacement, salvage), compute per-route TCO labels, then train seven classifiers to predict parity. Realistic field-noise is added so the problem is non-trivial. Reports full metric suite: accuracy, precision, recall, specificity, F1, ROC-AUC, plus PR curves and confusion matrices.

**Real data source pointers:** [WRI India e-bus TCO report](https://wri-india.org/publication/procurement-electric-buses-insights-total-cost-ownership-analyses) · [ICCT 2025 e-bus deployment report](https://theicct.org/publication/insights-from-public-deployment-and-case-study-of-private-intercity-operators-aug25/)

### 3. Segmenting Traffic Analysis Zones (Unsupervised)

The classical 4-stage transport model needs zone-type-specific trip generation coefficients. We apply unsupervised clustering to socio-economic TAZ data to find segments objectively rather than by planner judgement. K-Means is selected via elbow-plus-silhouette; we compare against Hierarchical (Ward), DBSCAN, and GMM, and visualise in 2D using PCA. Cluster profiles are presented as z-scores against the city mean, so a planner can interpret and name them.

**Real data source pointers:** [Census of India](https://censusindia.gov.in/) · [WRI CMP guides](https://wri-india.org/sustainable-cities) · [ITDP planning guides](https://www.itdp.org/library/standards-and-guides/)

## Running the notebooks

### Option A — Google Colab (no setup)

1. Click any notebook on GitHub → "Open in Colab"
2. Run all cells (`Runtime → Run all`)
3. The synthetic data CSVs ship with the repo; the notebooks read them via relative path

### Option B — Local Jupyter

```bash
git clone https://github.com/piyushjaangid/transport-ml-portfolio.git
cd transport-ml-portfolio
pip install -r dashboard/requirements.txt
jupyter notebook notebooks/
```

## Running the dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Open the printed local URL in a browser. To deploy free on Streamlit Cloud:

1. Push this repo to GitHub
2. Sign in at https://share.streamlit.io
3. New app → point at your `dashboard/app.py`
4. Done — the URL is yours to share on a CV

## Repo layout

```
transport-ml-portfolio/
├── README.md                       — this file
├── data/
│   ├── _generate_sample_data.py    — regenerates all CSVs from scratch
│   ├── case1_trips_hourly.csv      — Case 1 input
│   ├── case1_zones.csv             — Case 1 zone metadata
│   ├── case2_routes.csv            — Case 2 input
│   ├── case2_bus_specs.csv         — Case 2 bus specifications
│   ├── case2_operating_params.csv  — Case 2 yearly diesel/electricity prices
│   ├── case3_taz.csv               — Case 3 TAZ data
│   └── case3_skim.csv              — Case 3 origin-destination matrix
├── notebooks/
│   ├── 01_regression_trip_duration.ipynb
│   ├── 02_classification_ebus_tco.ipynb
│   └── 03_unsupervised_taz_clustering.ipynb
└── dashboard/
    ├── app.py                      — Streamlit app
    └── requirements.txt
```

## About me

**Piyush Jaangid** — M.Sc. Transport Planning & Engineering (Edinburgh Napier, UK), B.Tech Civil (IPU). Currently a Highway Engineer at Arjun Engineering & Consulting (Delhi), proof-checking 7 EPC mode National Highway projects for MoRTH and UP-PWD. GradCIHT, GMICE.

- 📧 PiyushJaangid@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/piyush-jaangid-089079190/) · [GitHub](https://github.com/piyushjaangid) · [Kaggle](https://www.kaggle.com/piyushjaangid) · [Blog](https://piyushjaangid.blogspot.com)

## License

MIT for the code. Data is synthetic and freely usable; real-data citations belong to their original publishers.
