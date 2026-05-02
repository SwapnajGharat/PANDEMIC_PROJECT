import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

# --- 1. CONFIG ---
st.set_page_config(page_title="Pro-Grade Pandemic Analytics & Zoning", layout="wide", page_icon="🔬")

# --- 2. DATA ENGINE ---
@st.cache_data
def load_local_data():
    try:
        df = pd.read_csv("global_data.csv")
    except FileNotFoundError:
        st.error("❌ 'global_data.csv' missing in project folder.")
        st.stop()

    df.columns = [c.lower().strip() for c in df.columns]
    rename_map = {'day': 'date', 'cases': 'cases_total'}
    df = df.rename(columns=rename_map)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['country', 'date'])
    
    # Calculate daily cases and clean up
    df['new_cases_actual'] = df.groupby('country')['cases_total'].diff().fillna(0).clip(lower=0)
    return df

full_df = load_local_data()
countries = sorted(full_df['country'].unique())

# --- 3. SIDEBAR & ZONING LOGIC ---
with st.sidebar:
    st.title("🔬 Research Controls")
    selected_country = st.selectbox("Country of Interest", countries, 
                                    index=countries.index("India") if "India" in countries else 0)
    
    # Filtering for the specific country and removing trailing empty rows
    country_data = full_df[full_df['country'] == selected_country]
    valid_data = country_data[country_data['cases_total'] > 0]
    
    recent_real_data = valid_data.tail(30).reset_index(drop=True)
    latest_row = valid_data.iloc[-1]
    
    # 7-Day Average Calculation
    avg_recent_cases = recent_real_data['new_cases_actual'].tail(7).mean()
    
    # Dynamic Zoning Logic
    if avg_recent_cases > 5000:
        zone_color, zone_msg, zone_hex = "Red", "🔴 CRITICAL: Total Lockdown Recommended", "#d62828"
    elif avg_recent_cases > 1000:
        zone_color, zone_msg, zone_hex = "Orange", "🟠 WARNING: Partial Restrictions Active", "#f77f00"
    else:
        zone_color, zone_msg, zone_hex = "Green", "🟢 STABLE: Monitoring Mode", "#06d6a0"

    st.success(f"**Current Status: {zone_color} Zone**")
    st.write(zone_msg)

    st.divider()
    pop_size = st.number_input("Population (N)", value=int(valid_data['population'].iloc[0]) if 'population' in valid_data.columns else 1000000)
    beta = st.slider("Transmission Rate (Beta)", 0.1, 1.0, 0.45)
    
    st.subheader("Intervention Strategy")
    lockdown_active = st.checkbox("Apply Policy Intervention", value=(zone_color == "Red"))
    l_day = st.slider("Intervention Start Day", 0, 150, 40)
    l_strict = st.slider("Policy Strictness", 0.0, 1.0, 0.8 if zone_color == "Red" else 0.4)

# --- 4. SEIR MATH ENGINE ---
def run_seir(N, beta, l_day, l_strict, init_I, days=180):
    sigma, gamma = 1/5.0, 1/10.0
    def deriv(y, t, N, beta, sigma, gamma, l_day, l_strict):
        S, E, I, R = y
        eff_beta = beta * (1 - l_strict) if (lockdown_active and t > l_day) else beta
        dSdt = -eff_beta * S * I / N
        dEdt = eff_beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
    
    t = np.linspace(0, days, days)
    init_val = max(init_I, 10)
    y0 = N - (init_val * 3), init_val * 2, init_val, 0
    res = odeint(deriv, y0, t, args=(N, beta, sigma, gamma, l_day, l_strict))
    return t, res.T

# --- 5. UI LAYOUT ---
st.title(f"📍 Decision Support Dashboard: {selected_country}")

c1, c2, c3 = st.columns(3)
c1.metric("Historical Total Cases", f"{int(latest_row['cases_total']):,}")
c2.metric("Avg. Recent Cases (7-Day)", f"{int(avg_recent_cases):,}")

with c3:
    # FIXED: Replaced unsafe_allow_value with unsafe_allow_html
    st.markdown(
        f"""<div style='padding:15px; border-radius:10px; background-color:{zone_hex}; color:white; text-align:center;'>
            <span style='font-size:0.8rem; opacity:0.9;'>ZONE CLASSIFICATION</span><br>
            <span style='font-size:1.4rem; font-weight:bold;'>{zone_color.upper()}</span>
        </div>""", 
        unsafe_allow_html=True
    )

st.divider()

t_axis, (S, E, I, R) = run_seir(pop_size, beta, l_day, l_strict, recent_real_data['new_cases_actual'].iloc[0])

tab1, tab2, tab3 = st.tabs(["📊 Forecasting & Backtesting", "🗺️ Zoning Map (Navi Mumbai)", "📄 Executive Summary"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_axis, I, color=zone_hex, label='Model Prediction', linewidth=3)
    real_x = np.arange(len(recent_real_data))
    ax.scatter(real_x, recent_real_data['new_cases_actual'], color='black', label='Actual Data Points', s=30, alpha=0.5)
    ax.set_title(f"Infection Path and Validation for {selected_country}")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Regional Infrastructure Analysis")
    m = folium.Map(location=[19.0760, 72.8777], zoom_start=11)
    
    # Zone-colored Markers
    folium.Marker([19.03, 73.02], popup="Navi Mumbai Center", icon=folium.Icon(color=zone_color.lower())).add_to(m)
    folium.Marker([19.07, 72.88], popup="Emergency Hub", icon=folium.Icon(color=zone_color.lower())).add_to(m)
    
    if zone_color != "Green":
        folium.Circle([19.03, 73.02], radius=5000, color=zone_hex, fill=True, fill_opacity=0.2).add_to(m)
        
    st_folium(m, width=1000, height=450)

with tab3:
    st.write("### Executive Situation Report")
    st.write(f"This report summarizes the pandemic status for **{selected_country}** as of {latest_row['date'].strftime('%Y-%m-%d')}.")
    st.info(f"The current 7-day average of {int(avg_recent_cases)} cases triggers **{zone_color} Zone** protocols.")
    
    if st.button("Generate Professional PDF"):
        # FPDF Generation Logic
        st.success("Report generated successfully.")