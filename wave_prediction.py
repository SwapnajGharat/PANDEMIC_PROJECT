import matplotlib
# THIS IS THE CRITICAL FIX: Tell Matplotlib to work without a window engine
matplotlib.use('Agg') 

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 1. The SEIR Differential Equations Logic
def deriv(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# 2. Setup Initial Parameters
N = 1000000             
E0, I0, R0 = 10, 1, 0   
S0 = N - E0 - I0 - R0   

# 3. Virus Dynamics
beta, sigma, gamma = 0.35, 1/5.0, 1/10.0 

# 4. Timeline
t = np.linspace(0, 160, 160) 

# 5. Solving
y0 = S0, E0, I0, R0
ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma))
S, E, I, R = ret.T

# 6. Data Science Insight
peak_infections = max(I)
peak_day = t[np.argmax(I)]

# 7. Visualization (Now using the 'Agg' backend)
plt.figure(figsize=(10, 6))
plt.plot(t, I, 'r', alpha=0.8, linewidth=2, label='Infected (Active Wave)')
plt.plot(t, E, 'y', alpha=0.8, linewidth=2, label='Exposed')
plt.plot(t, S, 'b', alpha=0.8, linewidth=2, label='Susceptible')
plt.axvline(x=peak_day, color='gray', linestyle='--', alpha=0.5)
plt.title('Pandemic Wave Forecast (SEIR Model)')
plt.xlabel('Days from Start')
plt.ylabel('Population Count')
plt.legend()
plt.grid(True)

# SAVE THE FILE
plt.savefig("wave_prediction.png")
print(f"SUCCESS: Graph saved as 'wave_prediction.png'. Peak Day: {int(peak_day)}")