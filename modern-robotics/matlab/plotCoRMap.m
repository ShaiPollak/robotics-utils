function plotCoRMap(contacts, x_range, y_range)
    % contacts: nx3 matrix [x, y, alpha]
    % x_range, y_range: plot limits [min, max]
    
    resolution = 150; 
    [X, Y] = meshgrid(linspace(x_range(1), x_range(2), resolution), ...
                      linspace(y_range(1), y_range(2), resolution));
    
    feasible_pos = ones(size(X)); % CCW rotation (+)
    feasible_neg = ones(size(X)); % CW rotation (-)
    n = size(contacts, 1);
    
    figure; hold on; grid on; axis equal;
    
    % --- Step 1: Calculate Grid Feasibility and Draw Sliding Lines ---
    for i = 1:n
        cx = contacts(i,1); cy = contacts(i,2); alpha = contacts(i,3);
        nx = cos(alpha); ny = sin(alpha);
        
        % Grid-based check for the colored regions
        v_dot_n_pos = -(cy - Y)*nx + (cx - X)*ny;
        feasible_pos = feasible_pos & (v_dot_n_pos >= -1e-9);
        
        v_dot_n_neg = (cy - Y)*nx - (cx - X)*ny;
        feasible_neg = feasible_neg & (v_dot_n_neg >= -1e-9);
        
        % Plot Sliding Lines (Normal lines)
        t = linspace(-20, 20, 100);
        plot(cx + t*nx, cy + t*ny, '--k', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    end
    
    % --- Step 2: Plot Feasible Regions (The "Clouds") ---
    contourf(X, Y, double(feasible_pos), [0.5 0.5], 'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'CCW CoR');
    contourf(X, Y, double(feasible_neg), [0.5 0.5], 'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'CW CoR');

    % --- Step 3: Mark ONLY Valid Rolling Points ---
    for i = 1:n
        ri = [contacts(i,1), contacts(i,2)];
        is_rolling_pos = true;
        is_rolling_neg = true;
        
        for j = 1:n
            if i == j, continue; end % Skip self-check
            
            nj = [cos(contacts(j,3)), sin(contacts(j,3))];
            rj = [contacts(j,1), contacts(j,2)];
            dist_vec = rj - ri;
            
            % Rotation about contact i: velocity at contact j is omega x (rj - ri)
            % For CCW (omega = +1): v = [-dy, dx]
            if dot([-dist_vec(2), dist_vec(1)], nj) < -1e-9
                is_rolling_pos = false;
            end
            % For CW (omega = -1): v = [dy, -dx]
            if dot([dist_vec(2), -dist_vec(1)], nj) < -1e-9
                is_rolling_neg = false;
            end
        end
        
        if is_rolling_pos || is_rolling_neg
            % Determine color based on which direction is valid
            p_color = 'y'; % Yellow if both
            if is_rolling_pos && ~is_rolling_neg, p_color = 'b'; end
            if ~is_rolling_pos && is_rolling_neg, p_color = 'r'; end
            
            plot(ri(1), ri(2), 'ko', 'MarkerFaceColor', p_color, 'MarkerSize', 8, 'DisplayName', 'Valid Rolling Pt');
        else
            % Invalid rolling point (Contact exists but blocks rotation)
            plot(ri(1), ri(2), 'kx', 'MarkerSize', 8, 'LineWidth', 1.5, 'HandleVisibility', 'off');
        end
    end

    % --- Step 4: Check for Linear Translation (Rotation at Infinity) ---
    angles = linspace(0, 2*pi, 17); % Check every 22.5 degrees
    for a = angles(1:end-1)
        v_lin = [cos(a), sin(a)];
        valid_lin = true;
        for j = 1:n
            nj = [cos(contacts(j,3)), sin(contacts(j,3))];
            if dot(v_lin, nj) < -1e-9
                valid_lin = false; break;
            end
        end
        if valid_lin
            quiver(mean(x_range), mean(y_range), v_lin(1)*3, v_lin(2)*3, ...
                'Color', [0 0.6 0], 'LineWidth', 2, 'MaxHeadSize', 1.5, 'DisplayName', 'Linear Escape');
        end
    end

    % Formatting
    title('Final CoR Map');
    subtitle('Yellow/Blue/Red Dots: Valid Rolling | X: Blocked Contact | Green: Linear Escape');
    xlabel('X'); ylabel('Y');
    xlim(x_range); ylim(y_range);
    legend('show', 'Location', 'northeastoutside');
end