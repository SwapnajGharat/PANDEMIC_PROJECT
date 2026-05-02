# 🛡️ Pandemic Intelligence & Zoning System

### **Overview**
A professional **Decision Support System (DSS)** built to forecast pandemic trends and automate regional safety protocols. This tool utilizes mathematical modeling to transform raw data into actionable zoning strategies.

---

### **🌟 Key Features**

*   **📈 SEIR Modeling:** Implements numerical integration of differential equations via `SciPy` to simulate and predict disease spread.
*   **🔴 Dynamic Zoning Engine:** Automatically categorizes regions into **Red, Orange, or Green zones** based on 7-day rolling averages and real-time case counts.
*   **📍 Geospatial Mapping:** Features an interactive visualization of healthcare infrastructure and regional risks in **Navi Mumbai** using `Folium`.
*   **⚙️ Live Data Management:** Includes a specialized CRUD (Create, Read, Update, Delete) module for manual data entry and error correction with permanent CSV updates.
*   **🧪 Predictive Backtesting:** A validation layer that compares theoretical SEIR models against historical data points to ensure forecast accuracy.

---

### **🛠️ Tech Stack**

*   **Language:** Python (Pandas, NumPy, SciPy)
*   **User Interface:** Streamlit
*   **Mapping:** Folium
*   **Modeling:** SEIR Framework (Susceptible-Exposed-Infectious-Recovered)

---

### **📈 System Preview**


<img width="1909" height="870" alt="Screenshot 2026-05-02 181328" src="https://github.com/user-attachments/assets/df049c19-3462-41e4-b7a5-d5b72446e457" />

<img width="1909" height="849" alt="Screenshot 2026-05-02 181345" src="https://github.com/user-attachments/assets/230560b0-a518-49dd-94c0-9cffce9f85ff" />



---

### **📂 Project Structure**
```text
├── ultra_app.py          # Main Streamlit Application
├── global_data.csv       # Persistent CSV Database
├── requirements.txt      # Project Dependencies
└── README.md             # Project Documentation
