classdef NMPC_System < matlab.System
    % NMPC System block implementing solve_nmpc function in Simulink
    
    properties (Nontunable)
        N = 20;         % Prediction horizon
        ts = 0.1;       % Sampling time
        Q_cost = diag([1, 1, 0.1]); % State cost
        R_cost = diag([0.01, 0.01]); % Input cost
        wrlmax = 5;     % Max wheel speed
        r = 0.1;        % Wheel radius
        d = 0.5;        % Distance between wheels
    end
    
    methods (Access = protected)
        
        function [x_opt, u_opt, wl_opt, wr_opt] = stepImpl(obj, Q, xr_h, yr_h, thetar_h)
            % Solve NMPC with variable-size signals
            [x_opt, u_opt, wl_opt, wr_opt] = solve_nmpc(...
                Q, xr_h, yr_h, thetar_h, obj.N, obj.ts, ...
                obj.Q_cost, obj.R_cost, obj.wrlmax, obj.r, obj.d ...
            );
        end
        
        function setupImpl(obj)
            % One-time setup for checking CasADi availability
            % Perform check only during simulation, not code generation
            if ~coder.target('MATLAB')
                if exist('casadi.Opti', 'class') ~= 8
                    error('CasADi is not installed or not in MATLAB path. Add CasADi to path.');
                end
            end
        end
        
        function num = getNumInputsImpl(obj)
            num = 4; % Q, xr_h, yr_h, thetar_h
        end
        
        function num = getNumOutputsImpl(obj)
            num = 4; % x_opt, u_opt, wl_opt, wr_opt
        end
        
        % Allow variable-size inputs by modifying this method
        function flag = isInputSizeMutableImpl(obj, inputPortIdx)
            % Enable variable-size input for all ports
            flag = true; % Allow variable-size signals on all input ports
        end
        
        % Optionally, define the maximum input size (if needed)
        function maxSize = getMaxInputSizeImpl(obj)
            % Set maximum size if required, otherwise leave as is
            maxSize = [Inf, Inf];  % Allow any size; modify this according to your needs
        end
        
        % Allow variable-size outputs (optional)
        function flag = isOutputSizeMutableImpl(obj)
            flag = true;  % Allow variable-size output signals
        end
    end
end
