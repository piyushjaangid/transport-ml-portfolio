"""
Streamlit Portfolio Dashboard — Piyush Jaangid
================================================
Three end-to-end ML case studies in transport / mobility.
Run locally:    streamlit run app.py
Deploy free:    https://share.streamlit.io
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
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CSS — make the selection buttons big, tappable, and visually distinct
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

div[data-testid="stButton"] > button {
    width: 100%;
    min-height: 180px;
    border: 2px solid #1F3864;
    border-radius: 12px;
    background: linear-gradient(135deg, #ffffff 0%, #f0f4fb 100%);
    color: #1F3864;
    font-size: 1.05rem;
    font-weight: 600;
    padding: 1.2rem 1rem;
    text-align: left;
    transition: all 0.18s ease;
    line-height: 1.5;
    white-space: normal;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #1F3864 0%, #2E75B6 100%);
    color: white;
    transform: translateY(-3px);
    box-shadow: 0 8px 18px rgba(31, 56, 100, 0.25);
    border-color: #1F3864;
}
div[data-testid="stButton"] > button:focus:not(:active) {
    border-color: #1F3864;
    box-shadow: 0 0 0 3px rgba(31, 56, 100, 0.2);
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #1F3864;
    margin-bottom: 0.3rem;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 0.8rem;
}
.hero-tagline {
    font-size: 0.95rem;
    color: #2E75B6;
    margin-bottom: 1.5rem;
    font-weight: 500;
}
.section-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1F3864;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1.5rem;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

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
# ROUTING via session state
# ----------------------------------------------------------------------------
PAGES = {
    "home": "🏠  Home",
    "case1": "🚕  1. Regression — Trip Duration",
    "case2": "🚌  2. Classification — E-Bus TCO",
    "case3": "🗺️  3. Unsupervised — TAZ Clustering",
}

if "page" not in st.session_state:
    st.session_state.page = "home"

def go(page_key: str):
    st.session_state.page = page_key

# ----------------------------------------------------------------------------
# SIDEBAR — backup navigation, plus author info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Piyush Jaangid")
    st.markdown(
        "**ML Portfolio — Transport & Mobility**\n\n"
        "Three end-to-end machine-learning case studies on real transport-sector problems."
    )

    st.markdown("---")
    st.markdown("**Navigate**")
    for key, label in PAGES.items():
        if st.button(label, key=f"sb_{key}", use_container_width=True):
            go(key)
            st.rerun()

    st.markdown("---")
    st.markdown(
        """**Links**
- [GitHub](https://github.com/piyushjaangid)
- [Kaggle](https://www.kaggle.com/piyushjaangid)
- [LinkedIn](https://www.linkedin.com/in/piyush-jaangid-089079190/)
- [Blog](https://piyushjaangid.blogspot.com)
        """
    )

page = st.session_state.page

# ============================================================================
# HOME PAGE — landing with prominent picker
# ============================================================================
if page == "home":
    st.markdown('<div class="hero-title">🚦 Transport-Sector ML Portfolio</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Piyush Jaangid &nbsp;·&nbsp; '
                'M.Sc. Transport Planning &nbsp;·&nbsp; B.Tech Civil</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-tagline">Three end-to-end ML case studies. '
                'Each follows the full 13-step pipeline. Pick one to begin →</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-label">Pick a case study</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        if st.button(
            "🚕  CASE 1\n\n"
            "Regression\n\n"
            "Predict ride-hail trip duration from booking-time features. "
            "Compares 7 algorithms — Linear → XGBoost.\n\n"
            "▸ MAE: ~13 s   |   R²: 0.96",
            key="card1",
            use_container_width=True,
        ):
            go("case1"); st.rerun()

    with col2:
        if st.button(
            "🚌  CASE 2\n\n"
            "Classification\n\n"
            "Predict whether an e-bus beats diesel on TCO over a 12-yr "
            "GCC contract. WRI / ICCT methodology.\n\n"
            "▸ F1: ~0.82   |   ROC-AUC: ~0.78",
            key="card2",
            use_container_width=True,
        ):
            go("case2"); st.rerun()

    with col3:
        if st.button(
            "🗺️  CASE 3\n\n"
            "Unsupervised\n\n"
            "Cluster Traffic Analysis Zones for the 4-stage transport model. "
            "K-Means, Hierarchical, DBSCAN, GMM.\n\n"
            "▸ Silhouette: > 0.30",
            key="card3",
            use_container_width=True,
        ):
            go("case3"); st.rerun()

    st.markdown("&nbsp;")
    st.markdown(
        "**Common methodology across all three:** "
        "Step 0 Problem definition → Step 1 Type selection → Step 2 Data → "
        "Step 3 EDA (correlation, multicollinearity pruning) → Steps 4–5 "
        "Cleaning, preprocessing, feature engineering → Steps 6–7 Train multiple "
        "algorithms → Step 8 Evaluate → Step 9 Cross-validate → Step 10 Tune → "
        "Step 11 Final test → Steps 12–13 Deploy & monitor."
    )

    st.markdown("&nbsp;")
    st.info(
        "💡 **For recruiters:** the case studies above are interactive — train "
        "models live, adjust sensitivity sliders, switch clustering algorithms. "
        "Source code and Jupyter notebooks are on "
        "[GitHub](https://github.com/piyushjaangid/transport-ml-portfolio)."
    )

# ============================================================================
# CASE 1 — REGRESSION
# ============================================================================
elif page == "case1":
    if st.button("← Back to home", key="back1"):
        go("home"); st.rerun()
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
elif page == "case2":
    if st.button("← Back to home", key="back2"):
        go("home"); st.rerun()
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
elif page == "case3":
    if st.button("← Back to home", key="back3"):
        go("home"); st.rerun()
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
