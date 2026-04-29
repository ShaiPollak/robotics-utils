#ifndef MR_ODOMETRY_H
#define MR_ODOMETRY_H

#include "mr_core.h"
#include <Eigen/Dense>



namespace mr {

namespace Odometry {

    /** @brief Compute the next state of the robot given the current configuration and control inputs
     * This function implements the odometry update for a drive robot (omnidirectional, mecanum, drift-drive). 
     * It takes into account the current orientation of the robot and the control inputs (wheel velocities) 
     * to compute the next configuration.
     * @param q_current Current configuration vector (size 3: phi, x, y)
     * @param u_controls Control input vector (size 3: d_phi, d_x, d_y)
     * @param dt Time step in seconds
     * @param F Transmission matrix that maps control inputs to body twist (H(0) dagger)
     * @return The next configuration vector
     */
    inline Eigen::VectorXd nextQState(const Eigen::VectorXd& q_current, const Eigen::VectorXd& u_controls, double dt, const Eigen::MatrixXd& F) {
        
        // q_current = [phi, x, y]
        double phi = q_current(0); 
        
        // *******Accumulated****** twist (distance) accum_Vb = [d_phi, d_x, d_y]
        Eigen::Vector3d accum_Vb = F * (u_controls * dt); 

        double dphi = accum_Vb(0);
        double dx = accum_Vb(1);
        double dy = accum_Vb(2);

        Eigen::Vector3d dq_body;

        if (std::abs(dphi) < nearly_zero_threshold) {
            // No Rotation, pure translation
            dq_body << 0, dx, dy;
        } else {
            // There is a rotation
            dq_body << dphi,
                    (dx * std::sin(dphi) + dy * (std::cos(dphi) - 1.0)) / dphi,
                    (dy * std::sin(dphi) + dx * (1.0 - std::cos(dphi))) / dphi;
        }

        // Rotation Matrix in the plane for the current orientation phi
        Eigen::Matrix3d Trans;
        Trans << 1, 0, 0,
                0, std::cos(phi), -std::sin(phi),
                0, std::sin(phi),  std::cos(phi);

        // Update the final state
        Eigen::VectorXd q_next = q_current + (Trans * dq_body);
        q_next(0) = mr::normalizeAngle(q_next(0)); // Normalize the angle to [-pi, pi]

        return q_next;
    }

        /**
     * @brief Updates the transformation matrix T_sb based on the change between q_current and q_next.
     * @param T_current The current 4x4 transformation matrix (Tsb_k)
     * @param q_current The current state vector [phi, x, y]
     * @param q_next    The next state vector [phi, x, y]
     * @return Eigen::Matrix4d The updated 4x4 transformation matrix (Tsb_k+1)
     */
    inline Eigen::Matrix4d nextOmniLocationInT(
        const Eigen::Matrix4d& T_current, 
        const Eigen::Vector3d& q_current, 
        const Eigen::Vector3d& q_next) {

        // 1. Calculate the change in the world frame
        double dphi = q_next(0) - q_current(0);
        double dx_world = q_next(1) - q_current(1);
        double dy_world = q_next(2) - q_current(2);

        // 2. Rotate the world-frame translation into the robot's CURRENT body frame
        // This gives us the displacement from the robot's perspective
        double phi = q_current(0);
        double dx_body = dx_world * std::cos(phi) + dy_world * std::sin(phi);
        double dy_body = -dx_world * std::sin(phi) + dy_world * std::cos(phi);

        // 3. Construct the local delta transformation (SE(2) update)
        Eigen::Matrix4d T_delta = Eigen::Matrix4d::Identity();

        if (std::abs(dphi) < nearly_zero_threshold) {
            // Case: Pure translation
            T_delta(0, 3) = dx_body;
            T_delta(1, 3) = dy_body;
        } else {
            // Case: Integrated arc motion (Exact integration)
            double s = std::sin(dphi);
            double c = std::cos(dphi);

            T_delta(0, 0) = c;   T_delta(0, 1) = -s;
            T_delta(1, 0) = s;    T_delta(1, 1) = c;

            T_delta(0, 3) = (dx_body * s + dy_body * (c - 1.0)) / dphi;
            T_delta(1, 3) = (dy_body * s + dx_body * (1.0 - c)) / dphi;
        }

        // 4. Update: T_next = T_current * T_delta
        // Post-multiplying applies the move in the LOCAL frame
        return T_current * T_delta;
    }
} // namespace Odometry
} // namespace mr

#endif
