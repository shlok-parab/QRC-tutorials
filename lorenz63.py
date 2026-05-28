import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz_63(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Classic Chaotic Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial conditions (x, y, z) and time span
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 50)
t_eval = np.linspace(0, 50, 5000)

# Solve the system
sol = solve_ivp(lorenz_63, t_span, initial_state,
                args=(sigma, rho, beta), t_eval=t_eval)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5, color='royalblue')

ax.set_title("Lorenz 63 Attractor")
ax.set_xlabel("X (Convection Rate)")
ax.set_ylabel("Y (Horizontal Temp Diff)")
ax.set_zlabel("Z (Vertical Temp Diff)")

plt.show()