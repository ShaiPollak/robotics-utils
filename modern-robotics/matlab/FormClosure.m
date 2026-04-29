function isFormClosure = FormClosure(setOfContacts)
    % setOfContacts: nx3 matrix where each row is [x, y, alpha]
    % x, y: coordinates of contact point
    % alpha: angle of the inward normal force (in radians)

    n = size(setOfContacts, 1);
    
    F_wrench_matrix = zeros(3, n);

    for i = 1:n
        x = setOfContacts(i, 1);
        y = setOfContacts(i, 2);
        alpha = setOfContacts(i, 3);
        
        % Create a normalized force
        fx = cos(alpha);
        fy = sin(alpha);
        
        % Moment cal
        mz = x*fy - y*fx;
        
        % Fill the Wrench Matrix
        F_wrench_matrix(:, i) = [mz; fx; fy];
    end

    % Check if matrix rank is at least 3
    if rank(F_wrench_matrix) < 3
        fprintf("The body is NOT in form closure, rank is less than 3 \n");
        isFormClosure = false;
        return;
    end 
    

    % Check if matrix has at least 4 col
    if n < 4
        fprintf("The body is NOT in form closure, we need at least 4 points of contact \n");
        isFormClosure = false;
        return;
    end
    
    % Prepare for optimization by defining the equality constraints
    f_obj = ones(n,1);
    Aeq = [F_wrench_matrix; ones(1,n)];
    beq = [zeros(3,1); 1];

    % Define lower and upper limit to the optimization problem
    lb = ones(n,1) * 0.001;
    ub = [];
    
    % Solve the optimization Problem
    options = optimoptions('linprog', 'Display', 'none');
    [k_sol, ~, exitflag] = linprog(f_obj, [], [], Aeq, beq, lb, ub, options);

    % 
    if exitflag == 1
        fprintf('The body is in FORM CLOSURE \n');
        disp('Found wrench magnitude (k): ');
        disp(k_sol);
        isFormClosure = true;
    else
        fprintf('The body is NOT in form closure. \n');
        isFormClosure = false;
    end

    try
        plotCoRMap(setOfContacts, [-10,10], [-10,10]);
    catch exception
        fprintf('Error displaying figure or plotCoRMap function missing \n')
        
    end
end
