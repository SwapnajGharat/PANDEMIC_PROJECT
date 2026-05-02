import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium

# --- STEP 1: GENERATE SAMPLE DATA ---
# We simulate a "Red Zone" cluster in a city and some isolated "Green" cases
np.random.seed(42)

# Create a dense cluster (The Hotspot)
hotspot_center = [19.0760, 72.8777] # Example: Mumbai area
hotspot_data = np.random.normal(loc=hotspot_center, scale=[0.005, 0.005], size=(60, 2))

# Create scattered cases (The Safe/Green areas)
scattered_data = np.random.normal(loc=hotspot_center, scale=[0.06, 0.06], size=(20, 2))

# Combine into one dataset
all_points = np.vstack([hotspot_data, scattered_data])
df = pd.DataFrame(all_points, columns=['lat', 'lon'])

# --- STEP 2: APPLY DATA SCIENCE (DBSCAN) ---
# eps: The distance (roughly 500m to 1km). 
# min_samples: How many people to form a 'Red Zone' cluster.
clustering = DBSCAN(eps=0.01, min_samples=10).fit(df[['lat', 'lon']])
df['cluster_id'] = clustering.labels_

# --- STEP 3: ASSIGN ZONES ---
# -1 means 'Noise' (Green), 0 or higher means 'Cluster' (Red)
def get_zone_color(label):
    if label == -1:
        return 'green'
    return 'red'

df['color'] = df['cluster_id'].apply(get_zone_color)

# --- STEP 4: CREATE INTERACTIVE MAP ---
m = folium.Map(location=hotspot_center, zoom_start=11)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        color=row['color'],
        fill=True,
        fill_opacity=0.7,
        popup=f"Zone: {'Red' if row['color'] == 'red' else 'Green'}"
    ).add_to(m)

# Save the map
m.save("pandemic_zones.html")
print("Project Created! Open 'pandemic_zones.html' in your browser to see the results.")