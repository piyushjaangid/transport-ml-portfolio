"""
Generates realistic sample datasets that MIMIC the structure of real public data.
Each notebook ships with these so it runs in <30 seconds on Colab without
needing users to download multi-GB files.

The notebooks include separate cells showing how to swap in the real datasets:
  - NYC TLC Trip Records (Parquet, ~50MB/month)
  - WRI/ICCT e-bus TCO inputs (manually transcribed from public reports)
  - OpenStreetMap + Census-style zonal data for 4-stage modelling
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("/home/claude/portfolio/data")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# =============================================================================
# CASE 1: Ride-hailing demand (Uber-style, NYC TLC schema)
# Real data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
# =============================================================================

# 30 days, 24 hours, 20 zones — gives 14,400 hourly aggregations
N_ZONES = 20
N_DAYS = 30
hours = pd.date_range("2024-09-01", periods=24 * N_DAYS, freq="h")

zones = pd.DataFrame({
    "zone_id": range(1, N_ZONES + 1),
    "zone_name": [f"Zone_{i:02d}" for i in range(1, N_ZONES + 1)],
    # Rough Delhi NCR lat/lng spread for plotting
    "lat": rng.uniform(28.45, 28.75, N_ZONES),
    "lng": rng.uniform(77.05, 77.40, N_ZONES),
    # Zone "type" drives demand patterns
    "zone_type": rng.choice(["CBD", "Residential", "Airport", "Mixed"],
                             N_ZONES, p=[0.2, 0.4, 0.05, 0.35]),
})
zones.to_csv(OUT / "case1_zones.csv", index=False)

# Build hourly demand with realistic daily + weekly seasonality
records = []
for ts in hours:
    hour = ts.hour
    dow = ts.dayofweek
    # Bimodal commute peaks weekdays, single late-night peak weekends
    if dow < 5:
        base_factor = 0.4 + 0.6 * np.exp(-((hour - 9) ** 2) / 8) + \
                      0.5 * np.exp(-((hour - 18) ** 2) / 6)
    else:
        base_factor = 0.5 + 0.7 * np.exp(-((hour - 22) ** 2) / 10)

    for _, z in zones.iterrows():
        zone_mult = {"CBD": 2.5, "Airport": 1.8, "Mixed": 1.2, "Residential": 0.8}[z.zone_type]
        # Weather noise — simulate occasional rain spikes
        weather = 1.0 + (0.4 if rng.random() < 0.05 else 0.0)
        # Surge: derived from supply/demand imbalance
        demand = max(1, int(rng.poisson(40 * base_factor * zone_mult * weather)))
        # Drivers available — lags demand
        drivers = max(1, int(rng.poisson(35 * base_factor * zone_mult * 0.85)))
        surge = max(1.0, min(3.5, demand / max(drivers, 1)))

        records.append({
            "pickup_datetime": ts,
            "zone_id": z.zone_id,
            "zone_type": z.zone_type,
            "trip_count": demand,
            "drivers_online": drivers,
            "avg_fare_inr": round(rng.normal(180, 40) * surge, 2),
            "avg_wait_time_sec": round(60 + 200 * (demand / max(drivers, 1)) + rng.normal(0, 15), 1),
            "surge_multiplier": round(surge, 2),
            "is_rain": int(weather > 1.0),
        })

df1 = pd.DataFrame(records)
df1.to_csv(OUT / "case1_trips_hourly.csv", index=False)
print(f"Case 1: {len(df1):,} rows across {N_ZONES} zones × {N_DAYS} days")

# =============================================================================
# CASE 2: E-bus TCO inputs
# Real source: WRI India "Procurement of Electric Buses" report (2021)
#              ICCT "Electrifying India's buses" (2025)
#              UITP/ICCT TCO methodology
# Numbers below are realistic order-of-magnitudes drawn from those reports.
# =============================================================================

# Bus configurations
bus_specs = pd.DataFrame([
    # bus_type, length_m, capex_lakh_inr, battery_kwh, energy_kwh_per_km,
    # range_km, mileage_kmpl_diesel, life_years, daily_km
    {"bus_type": "Diesel_9m_AC",     "capex_lakh": 80,    "battery_kwh": 0,    "energy_kwh_km": 0,    "diesel_kmpl": 4.2,  "life_yrs": 12, "daily_km": 200},
    {"bus_type": "Diesel_12m_AC",    "capex_lakh": 95,    "battery_kwh": 0,    "energy_kwh_km": 0,    "diesel_kmpl": 3.8,  "life_yrs": 12, "daily_km": 220},
    {"bus_type": "Electric_9m_AC",   "capex_lakh": 175,   "battery_kwh": 175,  "energy_kwh_km": 1.05, "diesel_kmpl": 0,    "life_yrs": 12, "daily_km": 200},
    {"bus_type": "Electric_12m_AC",  "capex_lakh": 220,   "battery_kwh": 250,  "energy_kwh_km": 1.30, "diesel_kmpl": 0,    "life_yrs": 12, "daily_km": 220},
])
bus_specs.to_csv(OUT / "case2_bus_specs.csv", index=False)

# Operating parameters (yearly time series for 12 yr horizon, MY 2024 base)
years = list(range(2024, 2024 + 13))
ops = pd.DataFrame({
    "year": years,
    # Diesel price growth ~5% nominal
    "diesel_price_inr_per_l": [95 * (1.05 ** i) for i in range(13)],
    # Electricity tariff for commercial chargers in INR/kWh (more stable)
    "electricity_inr_per_kwh": [9.5 * (1.025 ** i) for i in range(13)],
    # Maintenance cost INR/km
    "diesel_maint_inr_km": [4.5 * (1.04 ** i) for i in range(13)],
    "ebus_maint_inr_km": [2.2 * (1.04 ** i) for i in range(13)],
    # FAME-II / PM-eBus subsidy availability flag
    "subsidy_available": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})
ops.to_csv(OUT / "case2_operating_params.csv", index=False)

# Route-level data — 60 routes in a hypothetical city
# Tuned so the resulting TCO label is roughly balanced (~55-65% e-bus parity),
# matching the WRI sensitivity finding that gradient and utilisation drive the breakeven.
N_ROUTES = 60
routes = pd.DataFrame({
    "route_id": [f"R{i:03d}" for i in range(1, N_ROUTES + 1)],
    "route_km_oneway": rng.uniform(6, 22, N_ROUTES).round(1),
    "trips_per_day": rng.integers(4, 10, N_ROUTES),
    "avg_speed_kmh": rng.uniform(12, 32, N_ROUTES).round(1),
    # Wider gradient range — hilly routes (>4%) penalise e-bus efficiency
    "max_gradient_pct": rng.uniform(0.5, 8.0, N_ROUTES).round(2),
    "ridership_per_trip": rng.integers(20, 75, N_ROUTES),
})
routes.to_csv(OUT / "case2_routes.csv", index=False)
print(f"Case 2: {len(bus_specs)} bus types, {len(routes)} routes, {len(ops)} years of ops data")

# =============================================================================
# CASE 3: 4-stage transport model — zonal data + skim matrix
# Real source: any city's Comprehensive Mobility Plan; UK National Travel Survey;
#              India Census + economic surveys for trip rates
# =============================================================================

N_TAZ = 15  # Traffic Analysis Zones
taz = pd.DataFrame({
    "taz_id": range(1, N_TAZ + 1),
    "taz_name": [f"TAZ_{i:02d}" for i in range(1, N_TAZ + 1)],
    "population": rng.integers(15000, 95000, N_TAZ),
    "households": None,  # filled below
    "employment": rng.integers(3000, 60000, N_TAZ),
    "retail_employment": None,  # filled below
    "students_enrolled": rng.integers(2000, 25000, N_TAZ),
    # avg HH income in INR/month
    "median_hh_income": rng.integers(25000, 120000, N_TAZ),
    "lat": rng.uniform(28.50, 28.70, N_TAZ),
    "lng": rng.uniform(77.10, 77.35, N_TAZ),
})
taz["households"] = (taz["population"] / rng.uniform(3.8, 4.5, N_TAZ)).astype(int)
taz["retail_employment"] = (taz["employment"] * rng.uniform(0.15, 0.35, N_TAZ)).astype(int)
taz.to_csv(OUT / "case3_taz.csv", index=False)

# Skim matrices: travel time (min) and cost (INR) by mode
modes = ["car", "bus", "metro", "two_wheeler"]
records = []
for i in range(1, N_TAZ + 1):
    for j in range(1, N_TAZ + 1):
        if i == j:
            base_time = 5
            base_dist = 1.5
        else:
            # Use simple Euclidean on lat/lng
            d = np.sqrt(
                (taz.loc[i - 1, "lat"] - taz.loc[j - 1, "lat"]) ** 2 +
                (taz.loc[i - 1, "lng"] - taz.loc[j - 1, "lng"]) ** 2
            ) * 110  # rough km per degree
            base_dist = max(1.5, d)
            base_time = base_dist * 2.5  # ~24 km/h avg urban

        for m in modes:
            mult_t = {"car": 1.0, "bus": 1.6, "metro": 0.8, "two_wheeler": 0.95}[m]
            cost_per_km = {"car": 12, "bus": 2.5, "metro": 4, "two_wheeler": 4}[m]
            records.append({
                "origin": i, "destination": j, "mode": m,
                "travel_time_min": round(base_time * mult_t + rng.normal(0, 1), 2),
                "distance_km": round(base_dist, 2),
                "cost_inr": round(base_dist * cost_per_km, 2),
            })
skim = pd.DataFrame(records)
skim.to_csv(OUT / "case3_skim.csv", index=False)
print(f"Case 3: {N_TAZ} TAZs, skim matrix {len(skim):,} rows ({N_TAZ}×{N_TAZ}×{len(modes)} modes)")

print("\nAll sample data written to:", OUT)
print("\nFile sizes:")
for f in sorted(OUT.glob("*.csv")):
    print(f"  {f.name:<35} {f.stat().st_size/1024:.1f} KB")
