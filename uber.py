import joblib
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import h3

st.set_page_config(layout="wide", page_title="Uber H3 Surge Engine")

st.title("🚖 Uber H3 Spatial Surge & Demand Engine")
st.markdown("Real-time spatial demand forecasting and dynamic surge pricing using **Uber H3 Hexagonal Grids** and **XGBoost**.")

# 1. Load Data and Model Artifacts
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/h3_feature_matrix.csv')
    df['time_window'] = pd.to_datetime(df['time_window'])
    return df

@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_surge_model.pkl')

df = load_data()
model = load_model()

# 2. Sidebar Controls
st.sidebar.header("🕹️ Simulation Controls")

# Select Time Window
time_options = df['time_window'].dt.strftime('%Y-%m-%d %H:%M:%S').unique()
selected_time_str = st.sidebar.selectbox("Select Time Window:", time_options)

# Demand Shock Slider
demand_multiplier = st.sidebar.slider("Simulate Demand Shock (Multiplier):", 0.5, 3.0, 1.0, 0.1)

# Filter Data for Selected Window
filtered_df = df[df['time_window'] == pd.to_datetime(selected_time_str)].copy()

# Apply Demand Shock Simulation
filtered_df['lag_15m'] = filtered_df['lag_15m'] * demand_multiplier
filtered_df['neighborhood_demand'] = filtered_df['neighborhood_demand'] * demand_multiplier

# 3. Model Inference
features = ['hour', 'dayofweek', 'lag_15m', 'lag_60m', 'neighborhood_demand']
filtered_df['predicted_surge'] = model.predict(filtered_df[features])
filtered_df['predicted_surge'] = filtered_df['predicted_surge'].clip(lower=1.0)

# Convert H3 Cell IDs to Centroid Lat/Lng for PyDeck Mapping
def h3_to_lat_lng(hex_id):
    lat, lng = h3.cell_to_latlng(hex_id)
    return pd.Series([lat, lng])

filtered_df[['lat', 'lng']] = filtered_df['h3_hex'].apply(h3_to_lat_lng)

# Color Mapping: Green (1.0x) -> Yellow (1.3x) -> Red (1.5x+)
def get_color(surge):
    if surge > 1.4:
        return [239, 68, 68, 180]    # Red
    elif surge > 1.15:
        return [245, 158, 11, 180]   # Orange/Yellow
    return [34, 197, 94, 180]        # Green

filtered_df['color'] = filtered_df['predicted_surge'].apply(get_color)

# 4. Key Metrics Display
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Hexagons", len(filtered_df))
m2.metric("Avg Predicted Surge", f"{filtered_df['predicted_surge'].mean():.2f}x")
m3.metric("Peak Surge Hexagon", f"{filtered_df['predicted_surge'].max():.2f}x")
m4.metric("Demand Multiplier", f"{demand_multiplier}x")

st.markdown("---")

# 5. Render PyDeck 3D Map
st.subheader("📍 3D H3 Spatial Surge Heatmap")

view_state = pdk.ViewState(
    latitude=filtered_df['lat'].mean(),
    longitude=filtered_df['lng'].mean(),
    zoom=12,
    pitch=45
)

# PyDeck H3 Layer
layer = pdk.Layer(
    "H3HexagonLayer",
    filtered_df,
    get_hexagon="h3_hex",
    get_fill_color="color",
    get_elevation="predicted_surge * 100",
    elevation_scale=5,
    extruded=True,
    pickable=True,
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"html": "<b>Hex ID:</b> {h3_hex}<br/><b>Predicted Surge:</b> {predicted_surge}x<br/><b>Neighbor Demand:</b> {neighborhood_demand}"}
)

st.pydeck_chart(r)

# 6. Top High-Surge Table
st.subheader("🔥 Top 10 High-Demand Hexagons")
st.dataframe(
    filtered_df[['h3_hex', 'predicted_surge', 'demand_count', 'neighborhood_demand']]
    .sort_values(by='predicted_surge', ascending=False)
    .head(10),
    use_container_width=True
)