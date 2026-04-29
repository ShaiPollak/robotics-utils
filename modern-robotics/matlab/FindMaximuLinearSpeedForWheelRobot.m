function void = FindMaximuLinearSpeedForWheelRobot(H_matrix, u_max)

    H_linear = H_matrix(:, 2:3); 
    
    angles = linspace(0, 2*pi, 360);
    max_linear_speed = 0;
    
    for alpha = angles
        v_dir = [cos(alpha); sin(alpha)];
        
        u_required = H_linear * v_dir;
        
        scaling_factor = u_max / max(abs(u_required));
        
        current_speed = scaling_factor * norm(v_dir);
        
        if current_speed > max_linear_speed
            max_linear_speed = current_speed;
        end
    end
    
    fprintf('The maximum linear chassis speed is: %.2f\n', max_linear_speed);

end