function tau_history = runJointTrajectory(dataFile, durationHz, totalSeconds, startingConf, g, taulist, FTip)
    % RUNNEWTONEULERCONTROLLER Runs the N-E algorithm at a specific frequency.
    % 
    % Inputs:
    %   dataFile     - String, name of the .mat file (e.g., 'robotData.mat')
    %   durationHz   - Numeric, target frequency (e.g., 100)
    %   totalSeconds - Numeric, how long to run the loop
    %   startingConf - vector (numJoints)x1, theta_list of starting angles 
    %   eg [1; 1; 3.1; 3; 4; 5] for a 6R joint Robot
    %   g            - constant gravity vector g
    %   taulist      - constant vector of extra torque applied (Friction)
    %   FTip         - constant vector of extra force at the tip
    % Outputs:
    %   CSV file of [theta_list]s for each time step
    %   CSV file of [dtheta_list]s for each time step
    %   CSV file of [ddtheta_list]s for each time step

    

    
    %% 1. Load Data
    if isfile(dataFile)
        data = load(dataFile);
    else
        error('The data file %s was not found.', dataFile);
    end

    %% 2. Setup Timing and Pre-allocation
    dt = 1 / durationHz;
    numSteps = totalSeconds * durationHz;
    
    % Assuming numJoints is stored in your file, or detected from parameters
    [~, numJoints] = size(data.Slist);
    fprintf('Number of joints are: %d \n', numJoints);

    %Initializing forces
    if isempty(taulist)
        tau_applied = zeros(numJoints, 1);
    else
        tau_applied = taulist;
    end 
    
    if isempty(g)
        g_applied = [0; 0; -9.8];
    else
        g_applied = g;
    end

    if isempty(FTip)
        FTip_applied = zeros(6, 1);
    else
        FTip_applied = FTip;
    end

    %Initializing Theta lists
    theta = startingConf;
    if length(theta) ~= numJoints 
        error('Number of joints (%d), and number of thetas in theta_list (%d) are not compatible', numJoints, length(theta));
    end    
    dtheta = zeros(numJoints, 1);
    ddtheta = zeros(numJoints, 1);


    %Theta lists History Allocation
    tau_history = zeros(numJoints, numSteps);
    theta_history   = zeros(numSteps, numJoints);
    dtheta_history  = zeros(numSteps, numJoints);
    ddtheta_history = zeros(numSteps, numJoints);

    % Initialize the rate object for 100Hz
    r = rateControl(durationHz);

    %% 3. The Execution Loop
    fprintf('Executing Newton-Euler at %d Hz for %d seconds...\n', durationHz, totalSeconds);
 
    for k = 1:numSteps
        % --- Step A: Calculate Acceleration ---
        ddtheta = ForwardDynamics(theta, dtheta, tau_applied, g_applied, FTip_applied, data.Mlist, data.Glist, data.Slist);
        fprintf('ddTheta:\n [%s]\n', sprintf(' %.2f', ddtheta));


        fprintf('\n ------------------------------------- \n')
        time = k*dt;
        fprintf('Timestep: %d, Time: %.2f sec \n', k, time);

        % --- Step B: Store Current State ---
        theta_history(k, :)   = theta';
        dtheta_history(k, :)  = dtheta';
        ddtheta_history(k, :) = ddtheta';
        
        % --- Step C: Euler Integration ---
        % Update Velocity first, then Position (Semi-implicit Euler is more stable)
        
        theta = theta + dtheta*dt;
        dtheta = dtheta + ddtheta*dt;


        fprintf('Theta:\n [%s]\n', sprintf(' %.2f', theta));
        fprintf('dTheta:\n [%s]\n', sprintf(' %.2f', dtheta));
        
        

        % --- Step D: Sync to Clock ---
        waitfor(r);
    end
    
    fprintf('Writing data to CSV files...\n');
    writematrix(theta_history, 'theta_results.csv');
    writematrix(dtheta_history, 'dtheta_results.csv');
    writematrix(ddtheta_history, 'ddtheta_results.csv');
    
    fprintf('Loop finished successfully and files saved.\n');
end
