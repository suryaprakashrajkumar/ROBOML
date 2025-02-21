import numpy as np
import matplotlib.pyplot as plt

# Given controller parameters
K_P = 0.25
K_I = 0.75
K_FF = 0.15

# Simulation parameters
dt = 1/500  # Time step (s)
t_max = 5   # Simulation duration (s)
t = np.arange(0, t_max, dt)

# Desired command input (rad/s)
desired_command = np.ones((len(t), 2))  # Assuming a 2-wheel system

# Initialize system variables
error = np.zeros((len(t), 2))
integral = np.zeros((len(t), 2))
net_voltage = np.zeros((len(t), 2))
measurement = np.zeros((len(t), 2))

# Simulating the system response
for i in range(1, len(t)):
    error[i] = desired_command[i] - measurement[i-1]  # Compute error
    integral[i] = integral[i-1] + error[i] * dt  # Integral term
    
    # Compute control signal (PI + Feedforward)
    net_voltage[i] = K_P * error[i] + K_I * integral[i] + K_FF * desired_command[i]
    
    # Simulated system response (simplified model, assuming a first-order system)
    measurement[i] = measurement[i-1] + (net_voltage[i] - measurement[i-1]) * dt

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, desired_command[:, 0], label='Desired Command (rad/s)')
plt.plot(t, measurement[:, 0], label='Measured Speed (rad/s)', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Speed (rad/s)')
plt.title('PI Feedforward Controller Simulation')
plt.legend()
plt.grid()
plt.show()
