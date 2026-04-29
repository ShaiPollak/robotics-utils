function VisualizeBodyForceClosure(pairsOfContacts, CM, bodyToCheck)
    % Performs Moment Labeling (Reuleaux's Method) for a specific body.
    
    figure('Name', ['Force Closure: Body ', num2str(bodyToCheck)]); 
    hold on; grid on; axis equal;
    title(['Moment Regions: Body ', num2str(bodyToCheck)]);
    
    res = 250;
    [X_grid, Y_grid] = meshgrid(linspace(0,10,res), linspace(0,10,res));
    has_plus = false(size(X_grid));
    has_minus = false(size(X_grid));
    
    x_cm = CM(bodyToCheck, 1);
    y_cm = CM(bodyToCheck, 2);
    
    % Draw CM Reference Lines (Purple)
    h_cm_lines = plot([x_cm x_cm], [0 10], '--', 'Color', [0.5 0 0.5], 'LineWidth', 1);
    plot([0 10], [y_cm y_cm], '--', 'Color', [0.5 0 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');

    n_contacts = size(pairsOfContacts, 1);
    h_wrench = []; h_contact = [];

    for i = 1:n_contacts
        b1 = pairsOfContacts(i, 1);
        b2 = pairsOfContacts(i, 2);
        
        if b1 == bodyToCheck || b2 == bodyToCheck
            x_p = pairsOfContacts(i, 3);
            y_p = pairsOfContacts(i, 4);
            alpha = pairsOfContacts(i, 5);
            mu = pairsOfContacts(i, 6);
            beta = atan(mu);
            
            % NEWTON'S 3RD LAW: Flip wrench if checking the second body in the pair
            dir_mult = 1; 
            if b2 == bodyToCheck, dir_mult = -1; end
            
            rays = [alpha + beta, alpha - beta];
            for ang = rays
                fx = cos(ang) * dir_mult;
                fy = sin(ang) * dir_mult;
                
                % Moment M around grid points
                M = (X_grid - x_p) * fy - (Y_grid - y_p) * fx;
                has_plus = has_plus | (M > 1e-5);
                has_minus = has_minus | (M < -1e-5);
                
                % Plot Line of Action and Wrench Vector
                plot(x_p + [-20 20]*fx, y_p + [-20 20]*fy, ':', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
                h_wrench = quiver(x_p, y_p, fx, fy, 0, 'Color', [0 0.5 0], 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
            end
            h_contact = plot(x_p, y_p, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        end
    end
    
    % Moment Stacking Logic
    red_mask  = has_plus & ~has_minus;
    blue_mask = has_minus & ~has_plus;
    
    h_red = []; h_blue = [];
    if any(red_mask(:))
        [~, h_red] = contourf(X_grid, Y_grid, double(red_mask), [0.5 0.5], 'FaceColor', [1 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
    end
    if any(blue_mask(:))
        [~, h_blue] = contourf(X_grid, Y_grid, double(blue_mask), [0.5 0.5], 'FaceColor', [0.5 0.5 1], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
    end

    h_cm_point = plot(x_cm, y_cm, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
    
    % Assemble Legend
    handles = [h_wrench, h_contact, h_cm_point, h_cm_lines];
    labels = {'Applied Wrench', 'Contact Point', 'Body CM', 'CM Reference Lines'};
    if ~isempty(h_red), handles = [handles, h_red]; labels = [labels, 'Positive Moment (+) Only']; end
    if ~isempty(h_blue), handles = [handles, h_blue]; labels = [labels, 'Negative Moment (-) Only']; end
    
    xlabel('X [m]'); ylabel('Y [m]');
    legend(handles, labels, 'Location', 'northeastoutside');
    axis([0 10 0 10]);
end