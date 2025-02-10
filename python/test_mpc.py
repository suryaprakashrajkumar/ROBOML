import numpy as np
from casadi import Opti, vertcat

def solve_nmpc(Q, xr_h, yr_h, thetar_h, N, ts, Q_cost, R_cost, wrlmax, r, d):
    """
    Solves the NMPC optimization problem for a differential drive robot.
    
    Parameters:
        Q (numpy array): Initial state [x, y, theta].
        xr_h, yr_h, thetar_h (numpy arrays): Reference trajectory over the horizon.
        N (int): Prediction horizon.
        ts (float): Sampling time.
        Q_cost (numpy array): State cost matrix.
        R_cost (numpy array): Input cost matrix.
        wrlmax (float): Maximum wheel velocity.
        r (float): Wheel radius.
        d (float): Distance between wheels.
    
    Returns:
        numpy arrays: Optimized state trajectory x_opt, control inputs u_opt, left and right wheel speeds wl_opt, wr_opt.
    """
    opti = Opti()
    nx, nu = 3, 2  # State and control dimensions
    
    # Define variables
    x = opti.variable(nx, N + 1)
    u = opti.variable(nu, N)
    
    # Objective function and constraints
    objective = 0
    constraints = []
    v_min = -r * wrlmax
    v_max = r * wrlmax
    omega_min = -2 * wrlmax / d
    omega_max = 2 * wrlmax / d
    print(omega_max)
    print(v_max)
    print(wrlmax)
    for t in range(N):
        x_next = x[:, t] + ts * vertcat(
            u[0, t] * np.cos(x[2, t]),
            u[0, t] * np.sin(x[2, t]),
            u[1, t]
        )
        constraints.append(x[:, t + 1] == x_next)
        state_error = x[:, t] - vertcat(xr_h[t], yr_h[t], thetar_h[t])
        objective += state_error.T @ Q_cost @ state_error + u[:, t].T @ R_cost @ u[:, t]
        constraints.append(v_min <= u[0, t])
        constraints.append(u[0, t] <= v_max)
        constraints.append(u[1, t] >= omega_min)
        constraints.append(u[1, t] <= omega_max)
    
    # Terminal cost
    state_error = x[:, N] - vertcat(xr_h[N], yr_h[N], thetar_h[N])
    objective += state_error.T @ Q_cost @ state_error
    
    # Initial condition constraint
    constraints.append(x[:, 0] == Q)
    
    # Apply constraints to the optimization problem
    for c in constraints:
        opti.subject_to(c)
    
    # Set objective and solver
    opti.minimize(objective)
    opti.solver('bonmin')
    
    # Solve the optimization problem
    try:
        sol = opti.solve()
        x_opt = sol.value(x)
        u_opt = sol.value(u)
        
        # Compute wheel speeds
        wl_opt = (u_opt[0, :] - (d / 2) * u_opt[1, :]) / r
        wr_opt = (u_opt[0, :] + (d / 2) * u_opt[1, :]) / r
        
        return x_opt, u_opt, wl_opt, wr_opt
    except RuntimeError:
        print("Solver failed to find a solution.")
        return None, None, None, None

# Test the function
if __name__ == "__main__":
    # Define test parameters
    Q = np.array([0, 0, np.pi / 4])
    N = 25
    ts = 1 / 60
    Q_cost = np.diag([100, 100, 10])
    R_cost = np.diag([0.1, 0.1])
    r=0.04445        #radius od wheels [m]
    d= 0.393     #distance between the two wheels [m]
    wrwlmax=11 
    
    # Create a sample trajectory
    k = np.linspace(0, 2 * np.pi, N + 1)
    xr_h = 0.6 * np.sin(k)
    yr_h = 0.6 * np.cos(k)
    thetar_h = np.arctan2(yr_h, xr_h)
    
    # Solve NMPC
    x_opt, u_opt, wl_opt, wr_opt = solve_nmpc(Q, xr_h, yr_h, thetar_h, N, ts, Q_cost, R_cost, wrwlmax, r, d)
    
    print(wl_opt)
    # Store results individually
    # if x_opt is not None:
    #     np.save("x_opt.npy", x_opt)
    #     np.save("u_opt.npy", u_opt)
    #     np.save("wl_opt.npy", wl_opt)
    #     np.save("wr_opt.npy", wr_opt)
    #     print("Test completed successfully. Variables saved.")
    # else:
    #     print("Test failed. No solution found.")
