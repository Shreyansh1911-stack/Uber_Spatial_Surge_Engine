import os
import h3
import pandas as pd
import numpy as np

# Create output folders automatically if they don't exist
os.makedirs('data/processed', exist_ok=True)

print("Generating realistic NYC trip dataset...")

# Generate 50,000 simulated NYC ride pickups
np.random.seed(42)
n_samples = 50000

df = pd.DataFrame({
    'pickup_datetime': pd.date_range(start='2026-10-01', periods=n_samples, freq='5s'),
    'pickup_latitude': np.random.uniform(40.7000, 40.8000, n_samples),
    'pickup_longitude': np.random.uniform(-74.0200, -73.9300, n_samples),
})

# Filter out-of-bounds coordinates
df = df[
    (df['pickup_latitude'].between(40.50, 40.90)) & 
    (df['pickup_longitude'].between(-74.25, -73.70))
].copy()

# H3 Resolution 8 (~0.73 km2 per hexagon)
H3_RESOLUTION = 8

def assign_h3(row):
    # Updated to latlng_to_cell for h3 v4.x
    return h3.latlng_to_cell(row['pickup_latitude'], row['pickup_longitude'], H3_RESOLUTION)

print("Mapping Lat/Lng coordinates to Uber H3 Hexagons...")
df['h3_hex'] = df.apply(assign_h3, axis=1)

# Floor timestamps into 15-minute time windows
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['time_window'] = df['pickup_datetime'].dt.floor('15min')

# Aggregate demand per hexagon per 15-minute interval
spatial_demand = df.groupby(['h3_hex', 'time_window']).size().reset_index(name='demand_count')

# Save processed output directly into data/processed/
output_path = 'data/processed/h3_spatial_demand.csv'
spatial_demand.to_csv(output_path, index=False)

print(f"\nSUCCESS: Dataset created and saved to '{output_path}'!")
print(f"Total Rows: {len(spatial_demand)}")
print("\nFirst 5 rows of your processed CSV:")
print(spatial_demand.head())