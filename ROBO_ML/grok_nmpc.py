import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the differential drive robot
L = 0.3  # Distance between wheels (m)
dt = 0.1  # Sampling time (s)
N = 10   # Prediction horizon
v_max = 0.5  # Maximum linear velocity (m/s)
w_max = 1.0  # Maximum angular velocity (rad/s)

# State variables: x, y, θ
n_x = 3
# Input variables: v, ω
n_u = 2

# Define the system dynamics
def dynamics(x, u):
    x_next = x[0] + dt * u[0] * ca.cos(x[2])
    y_next = x[1] + dt * u[0] * ca.sin(x[2])
    theta_next = x[2] + dt * u[1]
    return ca.vertcat(x_next, y_next, theta_next)

# Define the cost function
def cost(x, u, x_ref):
    Q = ca.diag([10, 10, 1])  # Penalize position and orientation errors
    R = ca.diag([1, 1])       # Penalize control effort
    return (x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u

# Define the reference trajectory (simple circular path for demonstration)
def reference_trajectory(t):
    return ca.vertcat(ca.cos(t), ca.sin(t), 0)

# Setup optimization problem
opti = ca.Opti()

# Decision variables for states and controls over the horizon
X = opti.variable(n_x, N+1)
U = opti.variable(n_u, N)

# Initial condition
x0 = opti.parameter(n_x)
opti.subject_to(X[:, 0] == x0)

# Dynamics constraints
for k in range(N):
    opti.subject_to(X[:, k+1] == dynamics(X[:, k], U[:, k]))

# Control constraints
opti.subject_to(opti.bounded(-v_max, U[0, :], v_max))  # Linear velocity bounds
opti.subject_to(opti.bounded(-w_max, U[1, :], w_max))  # Angular velocity bounds

# Objective function
cost_obj = 0
for k in range(N):
    cost_obj += cost(X[:, k], U[:, k], reference_trajectory(k * dt))
cost_obj += cost(X[:, N], ca.DM.zeros(n_u), reference_trajectory(N * dt))  # Terminal cost
opti.minimize(cost_obj)

# Solver setup
opti.solver("ipopt")

# Simulation loop
x = np.array([0.0, 0.0, 0.0])  # Initial state
X_history = [x]
U_history = []
T = 10  # Total simulation time
t = 0

while t < T:
    # Set initial condition
    opti.set_value(x0, x)
    
    # Solve optimization problem
    sol = opti.solve()
    
    # Get the first control action
    u = sol.value(U[:, 0])
    U_history.append(u)
    
    # Apply control to simulate one step
    x_next = dynamics(x, u)
    x = x_next.full().flatten()
    X_history.append(x)
    
    t += dt

# Convert history to numpy arrays for plotting
X_history = np.array(X_history)
U_history = np.array(U_history)

# Plot results
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(X_history[:, 0], X_history[:, 1], label='Robot Path')
plt.plot(np.cos(np.arange(0, T, dt)), np.sin(np.arange(0, T, dt)), 'r--', label='Reference Path')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(0, T, dt), U_history[:, 0], label='Linear Velocity')
plt.plot(np.arange(0, T, dt), U_history[:, 1], label='Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Control Input')
plt.legend()
plt.tight_layout()
plt.show()