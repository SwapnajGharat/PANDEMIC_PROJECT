import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import folium
from streamlit_folium import st_folium
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pandemic Intelligence Dashboard", layout="wide")

# --- DATA LOADING ---
CSV_FILE = 'global_data.csv'

def load_data():
    if os.path.exists(CSV_FILE):
        data = pd.read_csv(CSV_FILE)
        # Ensure headers are clean (no hidden spaces)
        data.columns = [c.strip() for c in data.columns]
        return data
    else:
        st.error(f"Data file {CSV_FILE} not found!")
        return pd.DataFrame(columns=['Region', 'Confirmed', 'Recovered', 'Deaths', 'Latitude', 'Longitude'])

df = load_data()

# --- SEIR MODEL FUNCTION ---
def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# --- ZONING LOGIC ---
def calculate_zone(confirmed_cases):
    if confirmed_cases > 500:
        return "🔴 Red Zone", "#FF0000"
    elif confirmed_cases > 100:
        return "🟠 Orange Zone", "#FFA500"
    else:
        return "🟢 Green Zone", "#008000"

# --- SIDEBAR: DATA MANAGEMENT ---
st.sidebar.header("📊 Regional Data Management")
# Using 'Region' to match your new CSV header
selected_region = st.sidebar.selectbox("Select Region to Edit", df['Region'].unique())

col1, col2 = st.sidebar.columns(2)
with col1:
    add_cases = st.number_input("Add Cases", min_value=0, step=1)
with col2:
    remove_cases = st.number_input("Remove Cases", min_value=0, step=1)

if st.sidebar.button("Update CSV Database"):
    net_change = add_cases - remove_cases
    # Updating 'Confirmed' to match your new CSV header
    df.loc[df['Region'] == selected_region, 'Confirmed'] += net_change
    df.loc[df['Confirmed'] < 0, 'Confirmed'] = 0 
    
    df.to_csv(CSV_FILE, index=False)
    st.sidebar.success(f"Updated {selected_region} successfully!")
    st.rerun()

# --- MAIN DASHBOARD UI ---
st.title("🛡️ Pandemic Intelligence & Zoning System")

# Row 1: Metrics
total_cases = df['Confirmed'].sum()
st.metric("Total Confirmed Cases", f"{total_cases:,}")

# Row 2: Map and Model
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📍 Interactive Zoning Map")
    m = folium.Map(location=[19.0330, 73.0297], zoom_start=12)
    
    for index, row in df.iterrows():
        status, color = calculate_zone(row['Confirmed'])
        # Ensure your CSV has Latitude and Longitude columns!
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=10,
                popup=f"{row['Region']}: {row['Confirmed']} cases ({status})",
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)
    
    st_folium(m, width=800, height=500)

with right_col:
    st.subheader("📈 SEIR Prediction")
    N = 1000000
    t = np.linspace(0, 160, 160)
    y0 = (N-1, 1, 0, 0)
    res = odeint(seir_model, y0, t, args=(N, 0.3, 0.1, 0.2))
    plot_df = pd.DataFrame(res, columns=['S', 'E', 'I', 'R'])
    st.line_chart(plot_df[['I', 'E']])

# Row 3: Data Table
st.subheader("📋 Regional Statistics")
st.dataframe(df[['Region', 'Confirmed', 'Recovered', 'Deaths']], use_container_width=True)