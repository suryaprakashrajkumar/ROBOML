import numpy as np
from matplotlib import pyplot as plt
from casadi import SX, vertcat, Opti

# Parameters
ts = 1 / 30  # Sampling time
r = 0.0205  # Wheel radius
d = 0.053  # Distance between wheels
wrlmax = 10  # Maximum wheel speed (rad/s)

# Trajectory definition
eta = 0.6
alpha = 3.5
k = np.arange(0, 2 * np.pi * alpha * 2, ts)
xr = eta * np.sin(k / alpha)
yr = eta * np.sin(k / (2 * alpha))

# Velocity trajectory
xpr = eta * np.cos(k / alpha) * (1 / alpha)
ypr = eta * np.cos(k / (2 * alpha)) * (1 / (2 * alpha))

# Acceleration trajectory
xppr = -eta * np.sin(k / alpha) * (1 / alpha) ** 2
yppr = -eta * np.sin(k / (2 * alpha)) * (1 / (2 * alpha)) ** 2

# Driving velocity reference
vr = np.sqrt(xpr ** 2 + ypr ** 2)
wr = (yppr * xpr - xppr * ypr) / (xpr ** 2 + ypr ** 2)

# Orientation reference
thetar = np.arctan2(ypr, xpr)

# Adjust orientation
thetar_diff = np.diff(thetar)
i1, i2 = 0, len(thetar) - 1
for i in range(len(thetar_diff)):
    if thetar_diff[i] < -6:
        i1 = i + 1
    elif thetar_diff[i] > 6:
        i2 = i
thetar[i1:i2] += 2 * np.pi

thetar = np.unwrap(thetar)

# Initial conditions
x0, y0, theta0 = 0, 0, np.pi / 4
Q = np.array([x0, y0, theta0])

# NMPC parameters
N = 30  # Prediction horizon
nx = 3  # Number of states (x, y, theta)
nu = 2  # Number of inputs (linear and angular velocities)

# Define cost matrices
Q_cost = np.diag([500, 500, 10])
R_cost = np.diag([0.2, 0.2])

# Input constraints based on wheel speeds
v_min = -r * wrlmax
v_max = r * wrlmax
omega_min = - wrlmax / d
omega_max =  wrlmax *r / d

print(omega_max)
print(v_max)

# Initialize plot
plt.figure()
plt.plot(xr, yr, 'r--', label='Reference Trajectory', linewidth=1.5)
robot_path, = plt.plot(Q[0], Q[1], 'bo-', label='NMPC Trajectory', linewidth=0.5)
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('NMPC Trajectory Tracking with Live Plotting')
plt.grid()

# Store results
state_log = [Q]
control_log = []

# NMPC loop for the entire trajectory
for t_idx in range(len(k) - N):
    # Extract reference over the horizon
    xr_h = xr[t_idx:t_idx + N + 1]
    yr_h = yr[t_idx:t_idx + N + 1]
    thetar_h = thetar[t_idx:t_idx + N + 1]

    # Create optimization problem
    opti = Opti()

    # Define variables
    x = opti.variable(nx, N + 1)
    u = opti.variable(nu, N)

    # Objective and constraints
    objective = 0
    constraints = []

    for t in range(N):
        x_next = x[:, t] + ts * vertcat(
            u[0, t] * np.cos(x[2, t]),
            u[0, t] * np.sin(x[2, t]),
            u[1, t]
        )
        constraints.append(x[:, t + 1] == x_next)
        state_error = x[:, t] - vertcat(xr_h[t], yr_h[t], thetar_h[t])
        objective += state_error.T @ Q_cost @ state_error + u[:, t].T @ R_cost @ u[:, t]

        # Apply constraints on linear and angular velocities
        constraints.append(v_min <= u[0, t])
        constraints.append(u[0, t] <= v_max)
        constraints.append(u[1, t] >= omega_min)
        constraints.append(u[1, t] <= omega_max)

    # Terminal cost
    state_error = x[:, N] - vertcat(xr_h[N], yr_h[N], thetar_h[N])
    objective += state_error.T @ Q_cost @ state_error

    # Initial condition constraint
    constraints.append(x[:, 0] == Q)

    # Add constraints to the optimization problem
    for c in constraints:
        opti.subject_to(c)

    # Set objective
    opti.minimize(objective)

    # Solver settings
    opti.solver('bonmin')

    # Solve the optimization problem
    try:
        sol = opti.solve()
        x_opt = sol.value(x)
        u_opt = sol.value(u)

        # Update state and plot trajectory
        Q = x_opt[:, 1]
        state_log.append(Q)
        control_log.append(u_opt[:, 0])
        robot_path.set_xdata(np.append(robot_path.get_xdata(), Q[0]))
        robot_path.set_ydata(np.append(robot_path.get_ydata(), Q[1]))
        plt.draw()
        plt.pause(0.01)
    except RuntimeError:
        print("Solver failed to find a solution.")
        break

# Convert logs to arrays
state_log = np.array(state_log)
control_log = np.array(control_log)

# Plot control inputs
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(k[:len(control_log)], control_log[:, 0], label='$v$ (Linear Velocity)', color='b')
plt.plot(k, vr, '--', label='$v_r$ (Reference Linear Velocity)', color='r')
plt.legend()
plt.grid()
plt.ylabel('Linear Velocity [m/s]')

plt.subplot(2, 1, 2)
plt.plot(k[:len(control_log)], control_log[:, 1], label='$\omega$ (Angular Velocity)', color='b')
plt.plot(k, wr, '--', label='$\omega_r$ (Reference Angular Velocity)', color='r')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')

# Plot states vs reference
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(k[:len(state_log)], state_log[:, 0], label='$x$', color='b')
plt.plot(k, xr, '--', label='$x_r$', color='r')
plt.legend()
plt.grid()
plt.ylabel('x [m]')

plt.subplot(3, 1, 2)
plt.plot(k[:len(state_log)], state_log[:, 1], label='$y$', color='b')
plt.plot(k, yr, '--', label='$y_r$', color='r')
plt.legend()
plt.grid()
plt.ylabel('y [m]')

plt.subplot(3, 1, 3)
plt.plot(k[:len(state_log)], state_log[:, 2], label='$\\theta$', color='b')
plt.plot(k, thetar, '--', label='$\\theta_r$', color='r')
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Orientation [rad]')

plt.show()
