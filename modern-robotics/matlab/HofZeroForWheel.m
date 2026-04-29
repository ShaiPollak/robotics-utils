function HofZero = HofZeroForWheel(x, y, slid_ang, beta, r)
    % Returns a vector 
    % INPUTS:
    % x , y location of wheel center relative to robot's frame axis
    % slid_ang - angle of sliding (0 for omriwheel and pi/4 or -pi/4 for
    % mecanumwheel relative to the wheel's frame in radians
    % beta - angle of wheel frame relative to the robot's frame in radians
    % r - wheel's radius in meters

    %Rotation of wheel speed
    S = [1/r, tan(slid_ang)/r];

    % Rotation Matrix to adapt to wheel's frame
    R = [cos(beta)  sin(beta); 
        -sin(beta)  cos(beta)];

    % Location of center of wheel matrix
    T = [-y 1 0; 
          x 0 1];

    % Calculation
    HofZero = S * (R * T);
end