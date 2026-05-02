import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Pandemic Intelligence Pro", layout="wide")
plt.style.use('ggplot') 

# --- 2. THE ANALYTICS ENGINES ---

def run_seir_model(N, beta, days):
    """Predicts the wave using SEIR math."""
    sigma, gamma = 1/5.0, 1/10.0 
    def deriv(y, t, N, beta, sigma, gamma):
        S, E, I, R = y
        return -beta*S*I/N, beta*S*I/N - sigma*E, sigma*E - gamma*I, gamma*I
    
    t = np.linspace(0, days, days)
    y0 = N-11, 10, 1, 0
    ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma))
    return t, ret.T

def generate_zoning_map(points=100):
    """Calculates clusters for the map."""
    center = [19.0760, 72.8777]
    data = np.random.normal(loc=center, scale=[0.01, 0.01], size=(points, 2))
    df = pd.DataFrame(data, columns=['lat', 'lon'])
    clustering = DBSCAN(eps=0.005, min_samples=5).fit(df)
    df['zone'] = ['Red' if l != -1 else 'Green' for l in clustering.labels_]
    return df, center

# --- 3. THE UI LAYOUT ---
st.title("🛡️ Pandemic Intelligence & Response System")
st.markdown("Designed for Health Officials to simulate and monitor outbreaks.")

# SIDEBAR: Simulation Controls
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    transmission_rate = st.slider("Transmission Rate (Beta)", 0.1, 0.9, 0.35, 
                                  help="Low = Lockdown | High = Normal Spread")
    pop_size = st.number_input("City Population", value=1000000)
    
    st.markdown("---")
    st.header("👤 Quick Risk Check")
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    if st.button("Analyze Risk"):
        if fever and cough: st.error("High Risk - Isolate Immediately")
        else: st.success("Low Risk - Continue Monitoring")

# MAIN DASHBOARD TABS
tab1, tab2 = st.tabs(["📈 Epidemic Forecasting", "📍 Geospatial Zoning"])

with tab1:
    st.subheader("Future Wave Prediction")
    t, (S, E, I, R) = run_seir_model(pop_size, transmission_rate, 160)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, I, color='red', label='Active Infections')
    ax.fill_between(t, I, color='red', alpha=0.1)
    ax.set_xlabel("Days from Today")
    ax.set_ylabel("Infected Population")
    ax.legend()
    st.pyplot(fig)
    
    peak_day = int(t[np.argmax(I)])
    st.metric("Predicted Peak Day", f"Day {peak_day}", delta=f"{int(max(I))} cases")

with tab2:
    st.subheader("Hotspot Detection (DBSCAN)")
    df_zones, map_center = generate_zoning_map()
    m = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB positron")
    for _, row in df_zones.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color='red' if row['zone'] == 'Red' else 'green',
            fill=True
        ).add_to(m)
    st_folium(m, width=1000, height=500)

st.markdown("---")
st.caption("Built with Python, Scikit-Learn, and SEIR mathematical modeling.")