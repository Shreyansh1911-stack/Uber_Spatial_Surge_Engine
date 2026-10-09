import os
import h3
import pandas as pd
import numpy as np

print("Loading processed H3 spatial demand dataset...")
df = pd.read_csv('data/processed/h3_spatial_demand.csv')

# Ensure timestamp parsing
df['time_window'] = pd.to_datetime(df['time_window'])
df = df.sort_values(by=['h3_hex', 'time_window']).reset_index(drop=True)

# 1. Temporal Features
print("Generating temporal features...")
df['hour'] = df['time_window'].dt.hour
df['dayofweek'] = df['time_window'].dt.dayofweek

# 2. Temporal Lag Features per Hexagon
print("Computing 15-min and 60-min demand lags...")
df['lag_15m'] = df.groupby('h3_hex')['demand_count'].shift(1).fillna(0)
df['lag_60m'] = df.groupby('h3_hex')['demand_count'].shift(4).fillna(0)

# 3. Spatial Neighborhood Features (Adjacent H3 Hexagons)
print("Computing neighbor demand density across H3 k-rings...")

# Create a lookup mapping (h3_hex, time_window) -> demand_count for fast neighbor queries
demand_lookup = df.set_index(['h3_hex', 'time_window'])['demand_count'].to_dict()

def compute_neighborhood_demand(row):
    target_hex = row['h3_hex']
    current_time = row['time_window']
    
    # Get all 6 neighboring adjacent hexagons (k=1)
    neighbors = h3.grid_disk(target_hex, 1)
    
    # Sum demand across surrounding hexagons at the same time window
    total_neighbor_demand = 0
    for n in neighbors:
        total_neighbor_demand += demand_lookup.get((n, current_time), 0)
        
    return total_neighbor_demand

df['neighborhood_demand'] = df.apply(compute_neighborhood_demand, axis=1)

# 4. Target Variable Creation: Dynamic Surge Multiplier
# Top 15% high-demand time windows trigger surge pricing (>1.0x)
high_demand_threshold = df['demand_count'].quantile(0.85)

def calculate_surge(demand):
    if demand > high_demand_threshold:
        return round(1.0 + ((demand - high_demand_threshold) / 5.0), 2)
    return 1.0

df['surge_multiplier'] = df['demand_count'].apply(calculate_surge)

# Save Feature Matrix
os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/h3_feature_matrix.csv'
df.to_csv(output_path, index=False)

print(f"\nSUCCESS: Feature Matrix created and saved to '{output_path}'!")
print("\nFirst 5 rows of your feature dataset:")
print(df[['h3_hex', 'time_window', 'demand_count', 'lag_15m', 'neighborhood_demand', 'surge_multiplier']].head())