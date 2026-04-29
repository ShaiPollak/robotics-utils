function VisualizeGlobalAssembly(pairsOfContacts, CM)
    % Displays all bodies, their CMs, and the contact friction cones.
    
    figure('Name', 'Global Assembly View'); hold on; grid on; axis equal;
    title('Global Assembly: CMs and Contact Cones');
    
    num_bodies = size(CM, 1);
    n_contacts = size(pairsOfContacts, 1);
    g = 9.81;

    % Plot CM and Gravity for all bodies
    for j = 1:num_bodies
        x_cm = CM(j,1); y_cm = CM(j,2); mass = CM(j,3);
        plot(x_cm, y_cm, 'bo', 'MarkerFaceColor', 'b');
        text(x_cm, y_cm, ['  B' num2str(j)], 'Color', 'b', 'FontWeight', 'bold');
        quiver(x_cm, y_cm, 0, -mass*g*0.05, 0, 'Color', [0 0 0.6], 'LineWidth', 1.5);
    end

    % Plot all Friction Cones
    for i = 1:n_contacts
        x_p = pairsOfContacts(i,3); y_p = pairsOfContacts(i,4);
        alpha = pairsOfContacts(i,5); mu = pairsOfContacts(i,6);
        beta = atan(mu);
        
        rays = [alpha + beta, alpha - beta];
        for ang = rays
            quiver(x_p, y_p, cos(ang), sin(ang), 0, 'Color', [0 0.6 0], 'LineWidth', 1.2);
        end
        plot(x_p, y_p, 'rx', 'MarkerSize', 8, 'LineWidth', 1.5);
        text(x_p, y_p, sprintf(' C%d (%d:%d)', i, pairsOfContacts(i,1), pairsOfContacts(i,2)), 'FontSize', 8);
    end
    
    xlabel('X [m]'); ylabel('Y [m]');
    axis([0 10 0 10]);
end