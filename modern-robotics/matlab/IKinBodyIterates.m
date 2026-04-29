function [thetalist, success] = IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)
% *** CHAPTER 6: INVERSE KINEMATICS ***
% Takes Blist: The joint screw axes in the end-effector frame when the
%              manipulator is at the home position, in the format of a 
%              matrix with the screw axes as the columns,
%       M: The home configuration of the end-effector,
%       T: The desired end-effector configuration Tsd,
%       thetalist0: An initial guess of joint angles that are close to 
%                   satisfying Tsd,
%       eomg: A small positive tolerance on the end-effector orientation
%             error. The returned joint angles must give an end-effector 
%             orientation error less than eomg,
%       ev: A small positive tolerance on the end-effector linear position 
%           error. The returned joint angles must give an end-effector
%           position error less than ev.
% Returns thetalist: Joint angles that achieve T within the specified 
%                    tolerances,
%         success: A logical value where TRUE means that the function found
%                  a solution and FALSE means that it ran through the set 
%                  number of maximum iterations without finding a solution
%                  within the tolerances eomg and ev.
% Uses an iterative Newton-Raphson root-finding method.
% The maximum number of iterations before the algorithm is terminated has 
% been hardcoded in as a variable called maxiterations. It is set to 20 at 
% the start of the function, but can be changed if needed.  
% Example Inputs:
% 
% clear; clc;
% Blist = [[0; 0; -1; 2; 0; 0], [0; 0; 0; 0; 1; 0], [0; 0; 1; 0; 0; 0.1]];
% M = [[-1, 0, 0, 0]; [0, 1, 0, 6]; [0, 0, -1, 2]; [0, 0, 0, 1]];
% T = [[0, 1, 0, -5]; [1, 0, 0, 4]; [0, 0, -1, 1.6858]; [0, 0, 0, 1]];
% thetalist0 = [1.5; 2.5; 3];
% eomg = 0.01;
% ev = 0.001;
% [thetalist, success] = IKinBody(Blist, M, T, thetalist0, eomg, ev)
% 
% Output:
% thetalist =
%    1.5707
%    2.9997
%    3.1415
% success =
%     1

thetalist = thetalist0;
i = 0;
maxiterations = 20;
Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) * T));
err = norm(Vb(1: 3)) > eomg || norm(Vb(4: 6)) > ev;

theta_array = [thetalist'];

while err && i < maxiterations
    thetalist = thetalist + pinv(JacobianBody(Blist, thetalist)) * Vb;
    i = i + 1;
    
    %Calculate the end effector config, Twist Vb_se3 and Vb 
    T_current_body = FKinBody(M, Blist, thetalist);
    Vb_se3 = MatrixLog6(TransInv(T_current_body) * T);
    Vb = se3ToVec(Vb_se3);
    
    %Error Vector
    e_omg_i = norm(Vb(1: 3));
    e_v_i = norm(Vb(4: 6));

    %Printing Section:
    fprintf('Iteration %d: \n', i);
    fprintf('joint vector: %.2f %.2f %.2f %.2f %.2f %.2f\n', thetalist);
    fprintf('SE(3) end-effector config:\n');
    fprintf('%8.4f %8.4f %8.4f %8.4f\n', T_current_body');
    fprintf('error twist V_b: %.2f %.2f %.2f %.2f %.2f %.2f\n', Vb);
    fprintf('angular error magnitude ||omega_b||: %.2f\n', e_omg_i);
    fprintf('linear error magnitude ||v_b||: %.2f\n', e_v_i);
    fprintf('\n ---------------------------------------------- \n');

    %Check if passed threshold
    err = e_omg_i > eomg || e_v_i > ev;

    %Add thetalist to the theta array
    theta_array = [theta_array; thetalist'];



end
writematrix(theta_array, 'iterates.csv');
success = ~ err;
end