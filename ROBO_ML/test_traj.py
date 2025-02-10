import numpy as np
import matplotlib.pyplot as plt

def predict_waypoints(eta, alpha, ts, N, t_current):
    """
    Predicts waypoints over a horizon N based on the given trajectory equations.
    
    Parameters:
    eta (float): Scaling factor for trajectory.
    alpha (float): Trajectory shaping parameter.
    ts (float): Sampling time.
    N (int): Prediction horizon.
    t_current (float): Current time in the trajectory.
    
    Returns:
    tuple: Predicted waypoints (xr, yr, vr, wr, thetar).
    """
    k_end = t_current + N * ts
    k = np.arange(0, k_end, ts)
    
    # Position trajectory
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
    thetar = np.unwrap(thetar)
    
    # Find current index based on time
    k_current_idx = int(t_current / ts)
    k_future_idx = np.arange(k_current_idx, k_current_idx + N)
    
    return xr[k_future_idx], yr[k_future_idx], vr[k_future_idx], wr[k_future_idx], thetar[k_future_idx]

# Example usage
eta = 3.5
alpha = 5.5
ts = 1/60
N = 100  # Prediction horizon
t_current = 0.0  # Example current time
waypoints = predict_waypoints(eta, alpha, ts, N, t_current)

# Plotting results
xr, yr, _, _, _ = waypoints
plt.figure()
plt.plot(xr, yr, 'bo-', label='Predicted Waypoints')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Predicted Trajectory')
plt.legend()
plt.grid()
plt.show()

print(waypoints)
