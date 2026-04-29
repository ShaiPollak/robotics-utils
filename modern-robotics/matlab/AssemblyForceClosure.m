function isStable = AssemblyForceClosure(pairsOfContacts, CM)
    % ASSEMBLYFORCECLOSURE Determines if a multi-body assembly is stable under gravity.
    % 
    % This function checks for "Force Closure" by determining if there exists a 
    % set of contact forces within the friction cones that can balance the 
    % weight of all bodies in the assembly. It uses Linear Programming (linprog) 
    % to solve the equilibrium equations.
    %
    % INPUTS:
    %   pairsOfContacts: [n x 6] matrix where each row is a contact point:
    %       [body1_ID, body2_ID, x_pos, y_pos, normal_angle_rad, friction_coeff]
    %       - body_ID = 0 represents a fixed object (e.g., the ground).
    %       - normal_angle_rad is the angle of the surface normal.
    %   CM: [m x 3] matrix for Center of Mass properties of each body:
    %       [x_cm, y_cm, mass in kg]
    %       - Row index corresponds to the body ID.
    %
    % OUTPUT:
    %   isStable: Boolean (true if stable, false if it collapses).

    g = 9.81;
    n_contacts = size(pairsOfContacts, 1);
    num_bodies = max(max(pairsOfContacts(:, 1:2)));
    
    % Equilibrium matrix: 3 rows per body, 2 columns per contact
    A_eq = zeros(3 * num_bodies, 2 * n_contacts);
    b_eq = zeros(3 * num_bodies, 1);
    
    for i = 1:n_contacts
        b1 = pairsOfContacts(i, 1);
        b2 = pairsOfContacts(i, 2);
        x  = pairsOfContacts(i, 3);
        y  = pairsOfContacts(i, 4);
        alpha = pairsOfContacts(i, 5);
        mu = pairsOfContacts(i, 6);
        
        beta = atan(mu);
        
        % Compute the two friction cone rays (Wrenches)
        % Ray 1
        f1 = [cos(alpha + beta); sin(alpha + beta)];
        m1 = x*f1(2) - y*f1(1);
        w1 = [m1; f1(1); f1(2)];
        
        % Ray 2
        f2 = [cos(alpha - beta); sin(alpha - beta)];
        m2 = x*f2(2) - y*f2(1);
        w2 = [m2; f2(1); f2(2)];
        
        cols = [2*i-1, 2*i];
        
        % ADD to Body 1 equations
        if b1 > 0
            rows = (3*b1-2):(3*b1);
            A_eq(rows, cols) = [w1, w2];
        end
        
        % ADD to Body 2 equations (Negative sign for reaction)
        if b2 > 0
            rows = (3*b2-2):(3*b2);
            % IMPORTANT: We use the existing columns but subtract the wrench
            A_eq(rows, cols) = -[w1, w2];
        end
    end
    
    % Gravity: External Wrenches
    for i = 1:size(CM, 1)
        mass = CM(i, 3);
        x_cm = CM(i, 1);
        % Gravity wrench G = [m_z; f_x; f_y] = [x*(-mg); 0; -mg]
        G = [x_cm * (-mass * g); 0; -mass * g];
        
        % Equation is: A_eq * k + G = 0  =>  A_eq * k = -G
        rows = (3*i-2):(3*i);
        b_eq(rows) = -G;
    end

    % Solve LP
    f_obj = ones(2 * n_contacts, 1);
    lb = zeros(2 * n_contacts, 1);
    options = optimoptions('linprog','Display','none');
    
    [k_sol, ~, exitflag] = linprog(f_obj, [], [], A_eq, b_eq, lb, [], options);

    if exitflag == 1
        fprintf("Stable! Solution found. K sol: \n");
        disp(k_sol)
        isStable = true;
    else
        fprintf("Unstable. No solution satisfies constraints.\n");
        isStable = false;
    end

    RunFullAssemblyAnalysis(pairsOfContacts, CM);

end