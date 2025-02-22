load("Data\20d.mat")
% Extract data from Simulink 'out' variable
time = out.tout; % Extract time vector
xr = out.xr.Data; % Extract xr values
yr = out.yr.Data; % Extract yr values
thetar = out.thetar.Data; % Extract thetar values
u_opt = out.u_opt.Data; % Extract u_opt values
Q = out.Q.Data; % Extract Q values

% Get the number of time steps
numSteps = length(time);

% Initialize an empty matrix to store data
csvData = [];

% Loop through each time step
for i = 1:numSteps
    % Extract first 20 values of xr, yr, and thetar
    xr_20 = xr(1:20, i)';
    yr_20 = yr(1:20, i)';
    thetar_20 = thetar(1:20, i)';
    
    % Extract u_opt and Q for the current time step
    u_opt_values = [u_opt(1, 1:20, i), u_opt(2, 1:20, i)];
    Q_values = Q(i, :); % Convert 3×1 to 1×3
    
    % Concatenate time, extracted data, and control inputs
    rowData = [time(i), Q_values, xr_20, yr_20, thetar_20, u_opt_values];
    
    % Append to final data matrix
    csvData = [csvData; rowData];
end

% Define column headers
headers = ["time"];
% First, add all 20 xr values
% Finally, add Q values
headers = [headers, "Q1", "Q2", "Q3"];
for j = 1:20
    headers = [headers, "xr_" + j];
end
% Then, add all 20 yr values
for j = 1:20
    headers = [headers, "yr_" + j];
end
% Then, add all 20 thetar values
for j = 1:20
    headers = [headers, "thetar_" + j];
end
% Then, add u_opt1 values
for j = 1:20
    headers = [headers, "u_opt1_" + j];
end
% Then, add u_opt2 values
for j = 1:20
    headers = [headers, "u_opt2_" + j];
end

% Write to CSV file
csvFileName = 'D20.csv';
writematrix([headers; num2cell(csvData)], csvFileName);

disp('CSV file saved successfully.');
