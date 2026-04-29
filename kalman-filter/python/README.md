# In this filter kalman, we will be using the p_cov_check to UPDATE THE ERROR STATE!
# Then, after we calculate the error state, we will use it to UPDATE THE TRUE STATE!
# So, basicaly, there are two models:
# 1. The true physics model, which is the one we are trying to estimate, and is given by the process model.
# 2. The error state model, which is the one we are using to update our estimates, and is given by the error state dynamics.


#### 1. Model ##################################################################################


# The Model Provided here is as follows:
# State vector: x = [p, v, q]
# p is the position of the vehicle in the inertial frame, v is the velocity of the vehicle in the inertial frame, 
# and q is the orientation of the vehicle represented as a quaternion (in the inertial frame).
# 
# # The process model is given by:
# p_k = p_{k-1} + v_{k-1} * dt + 0.5 * R(q_{k-1}) * f_{k-1} * dt^2
# v_k = v_{k-1} + (R(q_{k-1}) * f_{k-1} + g) * dt
# q_k = q_{k-1} * q(omega_{k-1} * dt)
#
# which R(q) is the rotation matrix corresponding to the quaternion q, 
# and f_{k-1} is the specific force measurement from the IMU at time k-1.
# where g is the gravity vector, which we will assume to be 
# [0, 0, -9.81] m/s^2 in the inertial frame.
# omega_{k-1} is the angular velocity measurement from the IMU at time k-1, 
# aq(omega * dt) is the quaternion corresponding to the rotation of omega * dt radians

# Comparing to the book: Xk = F{k-1} * X{k-1} + G * u{k-1} + w{k-1}
# In our case, we have:
# Xk = [p_k, v_k, q_k]
# u{k-1} = [f_{k-1}, omega_{k-1}] "What is the "force" driving me forward? What is the "angular velocity" driving me forward?"
# w{k-1} is the process noise
#
# F{k-1} is the state transition matrix, It describes how a small change in the current state [p, v, q] affects the next state:
# F{k-1} = [[dp_k/dp_{k-1}, dp_k/dv_{k-1}, dp_k/dq_{k-1}],     [[I, dt*I, 0.5*R(q_{k-1})*f*dt^2/dq_{k-1}],
#           [dv_k/dp_{k-1}, dv_k/dv_{k-1}, dv_k/dq_{k-1}],  =   [0, I   , R(q_{k-1})*f*dt/dq_{k-1}      ],
#           [dq_k/dp_{k-1}, dq_k/dv_{k-1}, dq_k/dq_{k-1}]]      [0, 0   , I + Exp(omega*dt)             ]]
#
# G is the control input Jacobian matrix. It describes how your sensors/actuators [f, omega] (accelerometers and gyros) drive the state:
# G = [[dp_k/df_{k-1}, dp_k/domega_{k-1}],     [[0.5*R(q_{k-1})*dt^2, 0               ],
#      [dv_k/df_{k-1}, dv_k/domega_{k-1}],  =   [R(q_{k-1})*dt      , 0               ],
#      [dq_k/df_{k-1}, dq_k/domega_{k-1}]]      [0                  , q(Exp(omega*dt))]]
#
# Since we are dealing with rotations, this is not a linear system.
# We will need to use the Error State Extended Kalman Filter (ES-EKF) to handle the nonlinearity of the system.
# The error state is defined as:
# X_true = X_check + dx, where X_true is the true state, X_check is the predicted state, and dx is the error state.
# The error state for step k is defined as:
# dx_{k} = dx_{k-1} + (dx_dot_{k-1} * dt)
# According to the physics model, dx_dot can be derived from the process model and is given by:
# p_dot = v
# v_dot = R(q) * f + g
# q_dot = 1/2 * q * [0, omega]
# 
# Now we substitute the error state into the process model to get the error state dynamics:
# p_dot + dp_dot = v + dv
# v_dot + dv_dot = R(q + dq) * (f + df) + g = R(q) * f + R(q) * df + R(q) * skew[dtheta] * df + g
#
# Now we can linearize dx_dot = (True_Physics) - (Estimated_Physics):
# dp_dot = dv
# dv_dot = R(q) * df + R(q) * skew[dtheta] * f
# dtheta_dot = -skew[omega] * dtheta + domega
#
# The error for q hence to theta:
# q_true = q_check * dq = q_check * [1, 1/2 * dth]
# We remember that q_dot_true = q_dot_check * dq + q_check * dq_dot
# We subtitude q_dot into both the true and predicted state and multipling by the conjugate on the left:
# dq_dot = 1/2 (dq * [0, omega] - [0, omega] * dq) then subtituing omega = omega_check + domega and linearizing gives us:
# dtheta_dot = -skew[omega] * dtheta + domega

# Now we can write it in discrete time:
# dx_{k} = F{k-1} * dx_{k-1} + G{k-1} * w_{k}
#
# dp_{k} = dp_{k-1} + dv_{k-1} * dt - 1/2 * R(q) * skew[f] * dtheta * dt^2 
# dv_{k} = dv_{k-1} + (R(q) * df + R(q) * skew[dtheta] * f) * dt = dv_{k-1} + (R(q) * df - R(q) * skew[f] * dtheta) * dt
# dtheta_{k} = dtheta_{k-1} + (-skew[omega] * dtheta + domega) * dt
#
# For this case, F(9x9), L(9x6)  are:
# F{k-1} = [[I, dt*I, -0.5R*skew[f]*dt^2    ],
#           [0, I   , -R(q_{k-1})*skew[f]*dt],
#           [0, 0   , I - skew[omega]*dt    ]]
#
# L{k-1} =  [0            , 0               ],
#           [R(q_{k-1})*dt, 0               ],
#           [0            , I*dt            ]]


#### 2. Sensor Variances ########################################################################
# w ~ N(0, Q) where Q is the process noise covariance matrix
# w = [w_imu_f_x, w_imu_f_y, w_imu_f_z, w_imu_w_x, w_imu_w_y, w_imu_w_z]
# Hence Q(6x6) is:
# Q = [[var_imu_f, 0        , 0        , 0          , 0          , 0          ],
#      [0        , var_imu_f, 0        , 0          , 0          , 0          ],
#      [0        , 0        , var_imu_f, 0          , 0          , 0          ],
#      [0        , 0        , 0        , var_imu_w  , 0          , 0          ],
#      [0        , 0        , 0        , 0          , var_imu_w  , 0          ],
#      [0        , 0        , 0        , 0          , 0          , var_imu_w  ]]

#### 3. Measurement Models ###########################################################################
# The measurement model for the GNSS is given by:
# z_gnss = H_gnss * p + v_gnss
# where H_gnss is the measurement matrix for the GNSS, and v_gnss is the measurement noise for the GNSS.
# For the GNSS, we are measuring the **position** of the vehicle, so H_gnss is:
# H_gnss = [I, 0, 0]
# The measurement noise for the GNSS is given by:
# v_gnss ~ N(0, R_gnss) where R_gnss is the measurement noise covariance matrix for the GNSS, and is given by:
# R_gnss = [[var_gnss, 0       , 0       ],
#           [0       , var_gnss, 0       ],
#           [0       , 0       , var_gnss]]
# The measurement model for the LIDAR is given by:
# z_lidar = H_lidar * p + v_lidar
# where H_lidar is the measurement matrix for the LIDAR, and v_lidar is the measurement noise for the LIDAR.
# For the LIDAR, we are measuring the position of the vehicle, so H_lidar is:
# H_lidar = [I, 0, 0]
# The measurement noise for the LIDAR is given by:
# v_lidar ~ N(0, R_lidar) where R_lidar is the measurement noise covariance matrix for the LIDAR, and is given by:
# R_lidar = [[var_lidar, 0       , 0       ],
#            [0       , var_lidar, 0       ],
#            [0       , 0       , var_lidar]]