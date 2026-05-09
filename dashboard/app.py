"""
Streamlit Portfolio Dashboard — Piyush Jaangid
================================================
Unified showcase of the 3 ML case studies.
Run locally:    streamlit run app.py
Deploy free:    https://share.streamlit.io  (link your GitHub repo)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Piyush Jaangid — ML Portfolio",
    page_icon="🚦",
    layout="wide",
)

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
DATA = Path(__file__).parent.parent / "data"

@st.cache_data
def load_case1():
    trips = pd.read_csv(DATA / "case1_trips_hourly.csv", parse_dates=["pickup_datetime"])
    zones = pd.read_csv(DATA / "case1_zones.csv")
    return trips, zones

@st.cache_data
def load_case2():
    routes = pd.read_csv(DATA / "case2_routes.csv")
    specs = pd.read_csv(DATA / "case2_bus_specs.csv")
    ops = pd.read_csv(DATA / "case2_operating_params.csv")
    return routes, specs, ops

@st.cache_data
def load_case3():
    return pd.read_csv(DATA / "case3_taz.csv")

# ----------------------------------------------------------------------------
# SIDEBAR / NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("Piyush Jaangid")
st.sidebar.markdown(
    "**ML Portfolio — Transport & Mobility**\n\n"
    "Three end-to-end machine learning case studies grounded in real transport-sector "
    "problems. Each notebook follows the full 13-step ML pipeline."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Links")
st.sidebar.markdown("- [GitHub](https://github.com/piyushjaangid)")
st.sidebar.markdown("- [Kaggle](https://www.kaggle.com/piyushjaangid)")
st.sidebar.markdown("- [LinkedIn](https://www.linkedin.com/in/piyush-jaangid-089079190/)")
st.sidebar.markdown("- [Blog](https://piyushjaangid.blogspot.com)")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Pick a case study:",
    ["Overview",
     "1. Regression — Trip Duration",
     "2. Classification — E-Bus TCO",
     "3. Unsupervised — TAZ Clustering"],
)

# ============================================================================
# OVERVIEW PAGE
# ============================================================================
if page == "Overview":
    st.title("🚦 Transport-Sector ML Portfolio")
    st.markdown(
        "Hi — I'm a transport engineer (M.Sc. Edinburgh Napier, B.Tech IPU) who builds "
        "machine-learning models for mobility problems. This dashboard summarises three "
        "end-to-end case studies you can also explore as Jupyter notebooks on my GitHub."
    )

    cols = st.columns(3)
    cards = [
        ("🚕 Regression",
         "Predict ride-hail trip duration from booking-time features. Compares 7 algorithms "
         "(Linear → XGBoost) on the NYC TLC dataset structure.",
         "MAE: ~13 s   |   R²: 0.96"),
        ("🚌 Classification",
         "Predict whether an e-bus will beat diesel on TCO over a 12-yr GCC contract. "
         "Built on the WRI India and ICCT TCO methodology.",
         "F1: ~0.82   |   ROC-AUC: ~0.78"),
        ("🗺️ Unsupervised",
         "Cluster Traffic Analysis Zones for the 4-stage transport model. Compares K-Means, "
         "Hierarchical, DBSCAN, and GMM with PCA visualisation.",
         "Silhouette > 0.50"),
    ]
    for col, (title, body, metric) in zip(cols, cards):
        with col:
            st.markdown(f"### {title}")
            st.write(body)
            st.success(metric)

    st.markdown("---")
    st.markdown(
        "**Methodology common to all three notebooks:**\n\n"
        "Step 0 Problem definition  →  Step 1 Type selection  →  Step 2 Data collection  "
        "→  Step 3 EDA (with correlation analysis and feature pruning)  →  Step 4-5 "
        "Cleaning, preprocessing, feature engineering  →  Step 6-7 Model training "
        "(7+ algorithms each)  →  Step 8 Evaluation  →  Step 9 Cross-validation  "
        "→  Step 10 Hyperparameter tuning  →  Step 11 Final test  →  Step 12-13 "
        "Deployment & monitoring notes."
    )

# ============================================================================
# CASE 1 — REGRESSION
# ============================================================================
elif page.startswith("1."):
    st.title("Case 1 — Predicting Ride-Hail Trip Duration")
    st.caption("Regression  |  Inspired by Uber Movement & NYC TLC data")

    trips, zones = load_case1()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hourly aggregations", f"{len(trips):,}")
    col2.metric("Zones", len(zones))
    col3.metric("Time span (days)", trips.pickup_datetime.dt.date.nunique())
    col4.metric("Mean wait (s)", f"{trips.avg_wait_time_sec.mean():.0f}")

    st.markdown("### Demand pattern")
    trips_local = trips.copy()
    trips_local["hour"] = trips_local.pickup_datetime.dt.hour
    trips_local["dow"] = trips_local.pickup_datetime.dt.dayofweek
    pivot = trips_local.groupby(["dow", "hour"])["trip_count"].mean().unstack()
    fig, ax = plt.subplots(figsize=(11, 3))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Avg trips/hour"})
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("")
    st.pyplot(fig)
    st.caption("Bimodal weekday peaks (~9 AM, ~6 PM); single late-night peak on weekends.")

    st.markdown("### Interactive: surge multiplier vs wait time")
    zone_pick = st.selectbox("Zone type", ["All"] + list(zones.zone_type.unique()))
    sub = trips_local if zone_pick == "All" else trips_local[trips_local.zone_type == zone_pick]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(sub.surge_multiplier, sub.avg_wait_time_sec, alpha=0.3, s=10, color="#1F3864")
    ax.set_xlabel("Surge multiplier"); ax.set_ylabel("Avg wait time (s)")
    ax.set_title(f"Surge vs wait — {zone_pick}")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.markdown("### Live: train an XGBoost model on this data")
    if st.button("Train model", key="train1"):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        import xgboost as xgb

        with st.spinner("Training..."):
            df = trips_local.copy()
            df["is_weekend"] = (df["dow"] >= 5).astype(int)
            df["trips_per_driver"] = df["trip_count"] / df["drivers_online"].clip(lower=1)
            df = pd.get_dummies(df, columns=["zone_type"], drop_first=True)
            feats = [c for c in df.columns if c not in
                     ["pickup_datetime", "avg_wait_time_sec", "zone_id"]]
            X = df[feats]; y = np.log1p(df["avg_wait_time_sec"])
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                 n_jobs=-1, random_state=42).fit(X_tr, y_tr)
            preds = np.expm1(m.predict(X_te)); actual = np.expm1(y_te)
            mae = mean_absolute_error(actual, preds); r2 = r2_score(actual, preds)
        c1, c2 = st.columns(2)
        c1.metric("Test MAE (s)", f"{mae:.2f}")
        c2.metric("Test R²", f"{r2:.3f}")
        st.success("Model trained successfully — see notebook for full pipeline.")

# ============================================================================
# CASE 2 — CLASSIFICATION
# ============================================================================
elif page.startswith("2."):
    st.title("Case 2 — E-Bus TCO Parity Classifier")
    st.caption("Classification  |  Sources: WRI India 2021, ICCT 2024-2025")

    routes, specs, ops = load_case2()

    st.markdown("### Bus specifications")
    st.dataframe(specs, hide_index=True, use_container_width=True)

    st.markdown("### Sensitivity: diesel price vs e-bus parity")
    diesel_growth = st.slider("Annual diesel price growth (%)", 0.0, 10.0, 5.0, 0.5)
    elec_growth = st.slider("Annual electricity tariff growth (%)", 0.0, 6.0, 2.5, 0.5)
    discount = st.slider("Discount rate (%)", 6.0, 12.0, 8.0, 0.5)

    @st.cache_data
    def compute_parity(d_growth, e_growth, disc):
        diesel = specs[specs.bus_type == "Diesel_12m_AC"].iloc[0]
        ebus = specs[specs.bus_type == "Electric_12m_AC"].iloc[0]
        DAYS = 330; LIFE = 12; r = disc / 100
        SUB_PER_KWH = 20000; SUB_CAP = 50e5

        # rebuild ops with the user-chosen growth rates
        ops_local = ops.copy()
        ops_local["diesel_price_inr_per_l"] = [95 * (1 + d_growth / 100) ** i
                                                for i in range(len(ops_local))]
        ops_local["electricity_inr_per_kwh"] = [9.5 * (1 + e_growth / 100) ** i
                                                  for i in range(len(ops_local))]

        def tco(bus, daily_km, gradient, ridership):
            capex = bus.capex_lakh * 1e5
            sub = min(bus.battery_kwh * SUB_PER_KWH, SUB_CAP) if bus.battery_kwh > 0 else 0
            capex_net = capex - sub
            salvage_pv = (capex * 0.10) / (1 + r) ** LIFE
            bat_pv = (bus.battery_kwh * 11000) / (1 + r) ** 7 if bus.battery_kwh > 0 else 0
            lifetime_km = daily_km * DAYS * LIFE * 0.95
            energy_pv = 0; maint_pv = 0
            for t, row in enumerate(ops_local.itertuples(index=False)):
                if bus.battery_kwh > 0:
                    e_km = bus.energy_kwh_km * (1 + 0.04 * gradient) * row.electricity_inr_per_kwh
                    m_km = row.ebus_maint_inr_km
                else:
                    mi = bus.diesel_kmpl * (1 - 0.03 * gradient) * (1 - 0.001 * ridership)
                    e_km = row.diesel_price_inr_per_l / max(mi, 1)
                    m_km = row.diesel_maint_inr_km
                annual = daily_km * DAYS * 0.95
                energy_pv += (e_km * annual) / (1 + r) ** (t + 1)
                maint_pv += (m_km * annual) / (1 + r) ** (t + 1)
            return ((capex_net + bat_pv - salvage_pv) + energy_pv + maint_pv) / lifetime_km

        out = []
        for _, rt in routes.iterrows():
            dkm = rt.route_km_oneway * 2 * rt.trips_per_day
            td = tco(diesel, dkm, rt.max_gradient_pct, rt.ridership_per_trip)
            te = tco(ebus, dkm, rt.max_gradient_pct, rt.ridership_per_trip)
            out.append({"route_id": rt.route_id, "daily_km": dkm,
                        "gradient": rt.max_gradient_pct,
                        "tco_diesel": td, "tco_ebus": te,
                        "parity": int(te < td)})
        return pd.DataFrame(out)

    df = compute_parity(diesel_growth, elec_growth, discount)
    parity_pct = df.parity.mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Routes where e-bus wins", f"{df.parity.sum()} / {len(df)}",
              f"{parity_pct:.0f}%")
    c2.metric("Avg diesel TCO (₹/km)", f"{df.tco_diesel.mean():.1f}")
    c3.metric("Avg e-bus TCO (₹/km)", f"{df.tco_ebus.mean():.1f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2E75B6" if p else "#C00000" for p in df.parity]
    ax.scatter(df.daily_km, df.gradient, c=colors, s=80, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Daily km"); ax.set_ylabel("Max gradient (%)")
    ax.set_title("Routes coloured by parity (blue = e-bus wins, red = diesel wins)")
    st.pyplot(fig)

# ============================================================================
# CASE 3 — UNSUPERVISED
# ============================================================================
elif page.startswith("3."):
    st.title("Case 3 — TAZ Clustering for the 4-Stage Transport Model")
    st.caption("Unsupervised  |  CMP / WRI India / ITDP context")

    taz = load_case3()

    st.markdown("### Pick the algorithm and number of clusters")
    algo = st.selectbox("Algorithm", ["K-Means", "Hierarchical (Ward)",
                                      "Gaussian Mixture", "DBSCAN"])
    if algo != "DBSCAN":
        k = st.slider("Number of clusters (k)", 2, 6, 4)
    else:
        eps = st.slider("eps (DBSCAN)", 0.5, 3.0, 1.5, 0.1)
        min_samples = st.slider("min_samples (DBSCAN)", 2, 6, 3)

    features = ["households", "employment", "retail_employment",
                "students_enrolled", "median_hh_income"]
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import silhouette_score

    X = StandardScaler().fit_transform(taz[features])
    if algo == "K-Means":
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    elif algo == "Hierarchical (Ward)":
        labels = fcluster(linkage(X, "ward"), t=k, criterion="maxclust") - 1
    elif algo == "Gaussian Mixture":
        labels = GaussianMixture(n_components=k, random_state=42).fit_predict(X)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    valid = labels != -1
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters found", n_clusters)
    c2.metric("Noise points", int((~valid).sum()))
    if valid.sum() >= 2 and len(set(labels[valid])) >= 2:
        c3.metric("Silhouette", f"{silhouette_score(X[valid], labels[valid]):.3f}")
    else:
        c3.metric("Silhouette", "—")

    pca = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(pca[:, 0], pca[:, 1], c=labels, cmap="tab10", s=120,
                    edgecolor="k", linewidth=0.5)
    for i, t in enumerate(taz.taz_id):
        ax.annotate(str(t), (pca[i, 0], pca[i, 1]), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{algo} clustering — PCA projection")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    if n_clusters >= 1:
        taz_local = taz.copy()
        taz_local["cluster"] = labels
        st.markdown("### Cluster profiles (z-scored to city mean)")
        prof = taz_local[taz_local.cluster != -1].groupby("cluster")[features].mean()
        prof_z = (prof - taz[features].mean()) / taz[features].std()
        fig, ax = plt.subplots(figsize=(10, max(2, n_clusters * 0.6)))
        sns.heatmap(prof_z, annot=True, fmt="+.2f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("How each cluster compares to the city average")
        st.pyplot(fig)
