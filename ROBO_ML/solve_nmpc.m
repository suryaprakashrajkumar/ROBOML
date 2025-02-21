function [x_opt, u_opt, wl_opt, wr_opt] = solve_nmpc(Q, xr_h, yr_h, thetar_h, N, ts, Q_cost, R_cost, wrlmax, r, d)
    % NMPC solver using CasADi (compatible with Simulink Interpreted MATLAB Function Block)
    
    % Ensure CasADi is available
    if exist('casadi.Opti', 'class') ~= 8
        error('CasADi is not installed or not in MATLAB path. Add CasADi to path.');
    end

    % Create CasADi optimizer
    opti = casadi.Opti();

    % Define state and control dimensions
    nx = 3;  % [x, y, theta]
    nu = 2;  % [v, omega]

    % Decision variables
    x = opti.variable(nx, N+1);
    u = opti.variable(nu, N);

    % Cost function and constraints
    objective = 0;
    constraints = [];

    v_min = -r * wrlmax;
    v_max = r * wrlmax;
    omega_min = -2 * wrlmax / d;
    omega_max = 2 * wrlmax / d;

    for t = 1:N
        % Differential drive model
        x_next = x(:, t) + ts * [
            u(1, t) * cos(x(3, t));
            u(1, t) * sin(x(3, t));
            u(2, t)
        ];

        % Constraints
        constraints = [constraints, x(:, t+1) == x_next];

        % Cost function: State tracking + Control effort
        state_error = x(:, t) - [xr_h(t); yr_h(t); thetar_h(t)];
        objective = objective + state_error' * Q_cost * state_error + u(:, t)' * R_cost * u(:, t);

        % Input constraints
        constraints = [constraints, v_min <= u(1, t) <= v_max];
        constraints = [constraints, omega_min <= u(2, t) <= omega_max];
    end

    % Terminal cost
    state_error = x(:, N+1) - [xr_h(N+1); yr_h(N+1); thetar_h(N+1)];
    objective = objective + state_error' * Q_cost * state_error;

    % Initial condition constraint
    constraints = [constraints, x(:, 1) == Q];

    % Set constraints and objective
    opti.subject_to(constraints);
    opti.minimize(objective);

    % Set solver
    opti.solver('ipopt');

    % Solve optimization problem
    try
        sol = opti.solve();
        x_opt = full(sol.value(x));
        u_opt = full(sol.value(u));

        % Compute wheel speeds
        wl_opt = (u_opt(1, :) - (d / 2) * u_opt(2, :)) / r;
        wr_opt = (u_opt(1, :) + (d / 2) * u_opt(2, :)) / r;
    catch
        warning('Solver failed to find a solution.');
        x_opt = NaN(nx, N+1);
        u_opt = NaN(nu, N);
        wl_opt = NaN(1, N);
        wr_opt = NaN(1, N);
    end
end
