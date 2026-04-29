#ifndef MR_KINEMATICS_H
#define MR_KINEMATICS_H

#include <Eigen/Dense>
#include <cmath>
#include "mr_core.h"

namespace mr{
    
    namespace kinematics {

        inline double near_singularity_threshold = 1e-5; // Define a threshold for near-singularity

        // ROBOT JACOBIAN CALCULATIONS

        /** @brief Stacks two matrices horizontally (side by side).
         * * Logic & Formula:
         * 1. The resulting matrix C will have the same number of rows as A and B, and the number of columns will be the sum of the columns of A and B.
         * 2. The left part of C will be filled with the contents of A, and the right part will be filled with the contents of B.
         * 3. This is equivalent to concatenating the two matrices along their column dimension.
         * * @param A The first matrix to stack (m x p).
         * @param B The second matrix to stack (m x q).
         * @return A new matrix C (m x (p+q)) that is the horizontal stack of A and B.
         * * @usage_future
         * 1. Constructing Jacobians: When computing the Jacobian for a robot with both a mobile base and a manipulator arm, you often need to combine the Jacobian of the base (J_base) with the Jacobian of the arm (J_arm) to get the full Jacobian (J_e). This function can be used to stack J_base and J_arm horizontally to form J_e.
         * 2. Data Organization: In some algorithms, you may want to organize data from different sources into a single matrix for processing. For example, stacking sensor readings or control inputs together for batch processing.
         * 3. Feature Engineering: In machine learning applications, you might want to combine features from different sources into a single feature matrix for training a model.
         */
        inline Eigen::MatrixXd StackJacobians(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
            if (A.rows() != B.rows()) {
                throw std::invalid_argument("Matrices must have the same number of rows to stack horizontally.");
            }
            Eigen::MatrixXd C(A.rows(), A.cols() + B.cols());
            C << A, B;
            return C;
        }

        /** @brief Computes the 6xn Body Jacobian matrix for a serial chain robot.
         * * The Body Jacobian maps joint velocities (theta_dot) to the twist of the 
         * end-effector (Vb) frame expressed in end-effector frame {b}.
         * * Logic & Formula:
         * 1. The Jacobian columns are computed from the last joint back to the first.
         * 2. The n-th column (last joint) is always equal to the screw axis Bn itself:
         * Jb_n = Bn
         * 3. Each preceding column (i) is calculated by transforming the screw axis Bi
         * to the end-effector frame using the Adjoint map of the relative transformation:
         * Jb_i = Ad(T_i+1...n) * Bi 
         * * Iterative Algorithm:
         * - Start with T = Identity (4x4).
         * - For i from n-1 down to 0:
         * a. Set Jb.col(i) = Adjoint(T) * Blist.col(i)
         * b. Update T = T * MatrixExp6(-Blist.col(i) * thetalist(i)) 
         * (this is how joint i frame sees the end-effector pose by "walking back" through the kinematic chain).
         * (We use the negative twist to "walk back" from the end-effector to the base).
         * * @param Blist The 6xn matrix of screw axes in the body frame {b} at the home position.
         * @param thetalist A vector of current joint angles.
         * @return A 6xn matrix representing the Body Jacobian Jb(theta).
         * * @usage_future
         * 1. Inverse Kinematics: Used in the Newton-Raphson method to iteratively solve for joint angles that achieve a desired end-effector pose.
         * 2. Control: Used in resolved-rate control to compute joint velocities that achieve a
         * desired end-effector velocity in the body frame.
         * 3. Dynamics: Used in the computation of the mass matrix and Coriolis forces when expressed in the body frame.
         * 4. Redundancy Resolution: For robots with more degrees of freedom than necessary, the Jacobian can be used to find null-space motions that do not affect the end-effector pose.
         * 5. Singularities: Analyzing the Jacobian can help identify singular configurations where the robot loses degrees of freedom.
         *
         */
        inline Eigen::MatrixXd JacobianBody(const Eigen::MatrixXd& Blist, const Eigen::VectorXd& thetalist) {
            int n = thetalist.size();
            Eigen::MatrixXd Jb = Blist;
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            for (int i = n - 1; i >= 0; --i) {
                // Compute the i-th column of the Jacobian by applying the Adjoint transformation to the screw axis
                Jb.col(i) = Adjoint(T) * Blist.col(i);
                // Update the transformation T by "walking back" using the negative twist
                T = T * MatrixExp6(VecToSe3(-Blist.col(i) * thetalist(i)));
            }
            return Jb;
        }

        /** @brief Computes the 6xn Space Jacobian matrix for a serial chain robot.
         * * The Space Jacobian maps joint velocities (theta_dot) to the twist of the 
         * end-effector (Vs) frame expressed in (space) frame {s}.
         * * Logic & Formula:
         * 1. The Jacobian columns are computed from the first joint to the last.
         * 2. The i-th column (first joint) is always equal to the screw axis Si itself:
         * Js_i = Si
         * 3. Each subsequent column (i) is calculated by transforming the screw axis Si
         * to the end-effector frame using the Adjoint map of the relative transformation:
         * Js_i = Ad(T_1...i-1) * Si
         * * Iterative Algorithm:
         * - Start with T = Identity (4x4).
         * - For i from 1 to n-1:
         * a. Set Js.col(i) = Adjoint(T) * Slist.col(i)
         * b. Update T = T * MatrixExp6(Slist.col(i) * thetalist(i))
         * (this is how joint i frame sees the end-effector pose by "walking forward" through the kinematic chain).
         * (We use the positive twist to "walk forward" from the base to the end-effector).
         * * @param Slist The 6xn matrix of screw axes in the space frame {s} at the home position.
         * @param thetalist A vector of current joint angles.
         * @return A 6xn matrix representing the Space Jacobian Js(theta).
         * * @usage_future
         * 1. Inverse Kinematics: Used in the Newton-Raphson method to iteratively solve for joint angles that achieve a desired end-effector pose.
         * 2. Control: Used in resolved-rate control to compute joint velocities that achieve a desired end-effector velocity in the space frame.
         * 3. Dynamics: Used in the computation of the mass matrix and Coriolis forces when expressed in the space frame.
         * 4. Redundancy Resolution: For robots with more degrees of freedom than necessary, the Jacobian can be used to find null-space motions that do not affect the end-effector pose.
         * 5. Singularities: Analyzing the Jacobian can help identify singular configurations where the robot loses degrees of freedom. 
         */
        inline Eigen::MatrixXd JacobianSpace(const Eigen::MatrixXd& Slist, const Eigen::VectorXd& thetalist) {
            int n = thetalist.size();
            Eigen::MatrixXd Js = Slist;
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            for (int i = 1; i < n; ++i) {
                // Update the transformation T by "walking forward" using the twist
                T = T * MatrixExp6(VecToSe3(Slist.col(i - 1) * thetalist(i - 1)));
                // Compute the i-th column of the Jacobian by applying the Adjoint transformation to the screw axis
                Js.col(i) = Adjoint(T) * Slist.col(i);  
            }
            return Js;
        }

        // FORWARD AND INVERSE KINEMATICS

        /** @brief Computes the forward kinematics of a serial chain robot in the body frame.
         * * The function calculates the end-effector pose T in the body frame {b}
         *  given the home configuration M, the screw axes in the body frame Blist, and the current joint angles thetalist.
         * * Logic & Formula:
         * 1. Start with the home configuration M, which represents the end-effector pose
         * when all joint angles are zero.
         * 2. For each joint i, compute the transformation corresponding to that joint's motion
         * using the matrix exponential of the screw axis times the joint angle:
         * T_i = MatrixExp6(Blist.col(i) * thetalist(i))
         * 3. The overall transformation T is obtained by multiplying these transformations together in the order
         * they are applied, which is from the first joint to the last:
         * T_sb = M * T_1 * T_2 * ... * T_n
         * (Since we are working in the body frame, we multiply on the right).
         * * @param M The 4x4 home (all theta = 0) configuration matrix of the end-effector
         * @param Blist The 6xn matrix of screw axes in the body frame {b} at the home position.
         * @param thetalist A vector of current joint angles.
         * @return The 4x4 transformation matrix T representing the end-effector pose in
         * the body frame {b}.
         * * @usage_future
         * 1. Kinematics Simulation: To compute the end-effector pose given a set
         * of joint angles, which is essential for simulating the robot's motion.
         * 2. Control: To determine the current pose of the end-effector for feedback
         * control algorithms that require the end-effector position and orientation.
         * 3. Inverse Kinematics: Used in the Newton-Raphson method to iteratively solve for joint angles that achieve a desired end-effector pose by comparing the current pose to
         * the desired pose and computing the error twist.
         * 4. Trajectory Generation: To compute the end-effector pose at each point
         * along a trajectory defined in joint space, which can then be used to ensure the trajectory meets certain constraints or to visualize the motion.
         * 5. Dynamics: To compute the end-effector pose needed for calculating the mass matrix, Coriolis forces, and gravity forces when expressed in the body frame.
         */
        inline Eigen::Matrix4d FKinBody(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Blist, const Eigen::VectorXd& thetalist) {
            Eigen::Matrix4d T = M;
            int n = thetalist.size();
            for (int i = 0; i < n; ++i) {

                /** We multiply T on the right (because we're working in the body frame) by the transformation corresponding to joint i, 
                 * which is given by the matrix exponential of the screw axis times the joint angle. */
                T = T * MatrixExp6(VecToSe3(Blist.col(i) * thetalist(i)));
            }
            return T;
        }

        /** @brief Computes the forward kinematics of a serial chain robot in the space frame. 
         * * The function calculates the end-effector pose T in the space frame {s}
         * given the home configuration M, the screw axes in the space frame Slist,
         * and the current joint angles thetalist.
         * * Logic & Formula:
         * 1. Start with the home configuration M, which represents the end-effector pose
         * when all joint angles are zero.
         * 2. For each joint i, compute the transformation corresponding to that joint's motion
         * using the matrix exponential of the screw axis times the joint angle:
         * T_i = MatrixExp6(Slist.col(i) * thetalist(i))
         * 3. The overall transformation T is obtained by multiplying these transformations together in the order
         * they are applied, which is from the first joint to the last:
         * T_sb = T_1 * T_2 * ... * T_n * M
         * (Since we are working in the space frame, we multiply on the left).
         * * @param M The 4x4 home (all theta = 0) configuration matrix of the end-effector
         * @param Slist The 6xn matrix of screw axes in the space frame {s} at the home position.
         * @param thetalist A vector of current joint angles
         * @return The 4x4 transformation matrix T representing the end-effector pose in the space frame {s}.
         * * @usage_future
         * 
         */
        inline Eigen::Matrix4d FKinSpace(const Eigen::Matrix4d& M, const Eigen::MatrixXd& Slist, const Eigen::VectorXd& thetalist) {
            Eigen::Matrix4d T = M;
            int n = thetalist.size();
            for (int i = n - 1; i >= 0; --i) {
                /** We multiply T on the left (because we're working in the space frame) by the transformation corresponding to joint i, 
                 * which is given by the matrix exponential of the screw axis times the joint angle. */
                T = MatrixExp6(VecToSe3(Slist.col(i) * thetalist(i))) * T;
            }
            return T;
        }


        /** @brief Determines if a given Jacobian matrix is singular or near-singular.
         * * A Jacobian is singular if it loses rank, which typically occurs when the robot is in a configuration where it cannot move in certain directions (i.e., it has fewer degrees of
         * freedom than the number of joints). This can lead to issues in control and inverse kinematics, as it may not be possible to achieve certain end-effector velocities or poses.
         * * Logic & Formula:
         * 1. Compute the Singular Value Decomposition (SVD) of the Jacobian matrix J to obtain its singular values.
         * 2. The largest singular value (maxSigma) represents the maximum stretching factor of the Jacobian, while the smallest singular value (minSigma)
         * represents the minimum stretching factor. If minSigma is close to zero, it indicates that the Jacobian is losing rank and is therefore singular or near-singular.
         * 3. Compute the condition number of the Jacobian, which is the ratio of the largest singular value to the smallest singular value (conditionNumber = maxSigma / minSigma). A
         * high condition number (e.g., greater than 1000) is often used as a threshold to indicate that the Jacobian is near-singular, meaning that the robot is in a configuration where it has very limited mobility in certain directions.
         * * @param J The Jacobian matrix to be analyzed.
         * @return A boolean value indicating whether the Jacobian is singular or near-singular (true) or not (false).
         * * @usage_future
         * 1. Inverse Kinematics: To check if the Jacobian is singular before attempting to compute joint velocities from end-effector velocities, which can help avoid numerical issues.
         * 2. Control: To determine if the robot is in a configuration where it may have limited ability to move in certain directions, which can inform control strategies or trigger warnings.
         * 3. Trajectory Planning: To avoid planning trajectories that pass through singular configurations, which can lead to unpredictable behavior or loss of control.    * 
         */
        inline bool isSingular(const Eigen::MatrixXd& J) {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
            const auto& s = svd.singularValues();

            // Biggest singular value (the widest axis of the Jacobian ellipsoid)
            double maxSigma = s(0);
            // Smallest singular value (the narrowest axis of the Jacobian ellipsoid)
            double minSigma = s(s.size() - 1);

            // Numerical safeguard: if the smallest singular value is zero, the system is singular
            if (minSigma < near_singularity_threshold) {
                return true;
            }

            double conditionNumber = maxSigma / minSigma;

            // 1000 is a common threshold for considering a system to be near-singular. 
            // Adjust as needed for specific applications.
            return (conditionNumber > 1000.0);
        }

        /** @brief Computes the inverse kinematics of a serial chain robot in the body frame using the Newton-Raphson method.
         * * The function iteratively solves for the joint angles (thetalist) that achieve a desired end-effector pose T in the body frame {b}, given the home configuration M,
         * the screw axes in the body frame Blist, an initial guess for the joint angles thetalist0, and error tolerances for angular and linear components of the end-effector twist.
         * * Logic & Formula:
         * 1. Start with an initial guess for the joint angles (thetalist0)
         * 2. Compute the current end-effector pose using forward kinematics: T_current = FKinBody(M, Blist, thetalist)
         * 3. Compute the error twist Vb that represents the difference between the current end-effector pose and the desired pose T: Vb = Se3ToVec(MatrixLog6(
         * TransInv(T_current) * T))
         * 4. Check if the error is within the specified tolerances (eomg for angular error and ev for linear error). If it is, we have found a solution.
         * 5. If the error is not within tolerances, compute the Body Jacobian Jb at the current joint angles: Jb = JacobianBody(Blist, thetalist
         * 6. Compute the change in joint angles (delta_theta) using the pseudo-inverse of the Jacobian to solve for the desired end-effector twist: delta_theta = Jb^+ * Vb
         * (where Jb^+ is the pseudo-inverse of Jb, which can be computed using SVD or a more stable decomposition method).
         * 7. Update the joint angles: thetalist = thetalist + delta_theta
         * 8. Repeat steps 2-7 until the error is within tolerances or a maximum number of iterations is reached.
         * 
         * @param T The desired end-effector pose in the body frame {b} as a 4x4 transformation matrix.
         * @param M The 4x4 home (all theta = 0) configuration matrix of the end-effector.
         * @param Blist The 6xn matrix of screw axes in the body frame {b} at the home position.
         * @param thetalist0 An initial guess for the joint angles, which can affect convergence.
         * @param eomg The error tolerance for the angular component of the end-effector twist (in radians).
         * @param ev The error tolerance for the linear component of the end-effector twist (in meters).
         * @return A pair consisting of:
         * - A vector of joint angles (thetalist) that achieve the desired end-effector pose within the specified tolerances, or the best attempt if convergence was not achieved.
         * - A boolean value indicating whether the algorithm successfully converged to a solution within the error tolerances (
         * true) or if it failed to converge (false), which can occur if the maximum number of iterations is reached or if the Jacobian becomes singular during the process).
         * @usage_future
         * 1. Robotics Control: To compute the joint angles required to achieve a desired end-effector pose, which is essential for tasks such as pick-and-place, assembly, or any task that requires precise positioning of the end-effector.
         * 2. Simulation: To test and validate robot kinematics and control algorithms by computing
         * the joint angles for various desired end-effector poses and simulating the resulting motion.
         * 3. Trajectory Planning: To compute the joint angles along a trajectory defined in task space (end-effector pose) and ensure that the robot can follow the trajectory within specified error
         */
        inline std::pair<Eigen::VectorXd, bool> IKinBodyNR(const Eigen::Matrix4d& T, const Eigen::Matrix4d& M, 
            const Eigen::MatrixXd& Blist, const Eigen::VectorXd& thetalist0, double eomg, double ev) {
            Eigen::VectorXd thetalist = thetalist0;
            int maxiterations = 20; //Change this if you want to allow more iterations

            // First, we compute the forward kinematics to find the current 
            // end-effector pose given the initial guess (thetalist0) of joint angles.
            Eigen::Matrix<double, 6, 1> Vb = Se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) * T));
            bool err = (Vb.head<3>().norm() > eomg) || (Vb.tail<3>().norm() > ev);

            while (err && maxiterations > 0) {
                Eigen::MatrixXd Jb = JacobianBody(Blist, thetalist);
                
                if (isSingular(Jb)) {
                    // If the Jacobian is singular, we cannot proceed with the inverse kinematics solution.
                    return {thetalist, false};
                }

                /** We compute the change in joint angles (delta_theta) using the pseudo-inverse of the Jacobian 
                * to solve for the desired end-effector twist Vb.
                * colPivHouseholderQr() calculates P, Q and R such that A = Q * R * P^T,
                * where P is a permutation matrix, Q is an orthogonal matrix, and R is an upper triangular matrix.
                * solve() then solves the linear system A * x = b for x, 
                * where A is the matrix we want to invert (Jb) and b is the vector we want to match (Vb).
                * This handes cases where Jb is not square or is close to singular by using a more stable decomposition method.
                * If Jb is not full rank it will return a least-squares solution that minimizes the error ||Jb * delta_theta - Vb||.
                */

                thetalist = thetalist + Jb.colPivHouseholderQr().solve(Vb);
                Vb = Se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) * T));
                err = (Vb.head<3>().norm() > eomg) || (Vb.tail<3>().norm() > ev);

                maxiterations--;
            }
            return {thetalist, !err};
        }

        /** @brief Computes the inverse kinematics of a serial chain robot in the body frame using the Damped Least Squares (DLS) method.
         * * The Damped Least Squares method is an extension of the Newton-Raphson method that adds a damping term to the pseudo-inverse calculation to improve stability, especially near singular configurations.
         * * Logic & Formula:
         * 1. The overall structure of the algorithm is similar to the Newton-Raphson method, with the same steps for computing the current end-effector pose, error twist, and checking
         * error tolerances.
         * 2. The key difference is in the computation of the change in joint angles (delta_theta). Instead of using the standard pseudo-inverse, we compute a damped pseudo-inverse
         * of the Jacobian:
         * delta_theta = (J^T * J + lambda^2 * I)^(-1) * J^T * Vb
         * where lambda is a damping factor that can be tuned based on the specific robot and application. A common choice for lambda is a small positive value
         * (e.g., 0.01) that provides a balance between stability and convergence speed. The term lambda^2 * I adds a small value to the diagonal of J^T * J, which helps to prevent large changes in joint angles when the Jacobian is near-singular.
         * 3. The rest of the algorithm proceeds as in the Newton-Raphson method, with iterative updates to the joint angles and recomputation of the error twist until convergence or maximum iterations
         * @param T The desired end-effector pose in the body frame {b} as a 4x4 transformation matrix.
         * @param M The 4x4 home (all theta = 0) configuration matrix of the end-effector.
         * @param Blist The 6xn matrix of screw axes in the body frame {b} at the home position.
         * @param thetalist0 An initial guess for the joint angles, which can affect convergence.
         * @param eomg The error tolerance for the angular component of the end-effector twist (in radians).
         * @param ev The error tolerance for the linear component of the end-effector twist (in meters).
         * @return A pair consisting of:
         * - A vector of joint angles (thetalist) that achieve the desired end-effector pose within the specified tolerances, or the best attempt if convergence was not achieved.
         * - A boolean value indicating whether the algorithm successfully converged to a solution within the error tolerances (true) or if it failed to converge (false), which can occur if the maximum number of iterations is reached or if the Jacobian becomes singular during the process).
         * @usage_future - Same as the Newton-Raphson method, but with improved stability near singular configurations, making it more suitable for robots that may encounter such configurations during their operation.
         * 
         */
        inline std::pair<Eigen::VectorXd, bool> IKinBody(const Eigen::Matrix4d& T, const Eigen::Matrix4d& M, 
        const Eigen::MatrixXd& Blist, const Eigen::VectorXd& thetalist0, double eomg, double ev) {
        
            Eigen::VectorXd thetalist = thetalist0;
            int maxiterations = 50; 
            double lambda = 0.1; // Damping factor, adjust as needed

            Eigen::VectorXd Vb = Se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) * T));
            bool err = (Vb.head<3>().norm() > eomg) || (Vb.tail<3>().norm() > ev);

            while (err && maxiterations > 0) {
                Eigen::MatrixXd Jb = JacobianBody(Blist, thetalist);

                // Check for singularity before computing the DLS solution
                if (isSingular(Jb)) {
                    return {thetalist, false};
                }
                
                /**  We compute the change in joint angles (delta_theta) using the Damped Least Squares method to solve for the desired end-effector twist Vb.
                 *   The DLS method adds a damping term to the pseudo-inverse calculation to improve stability, especially near singular configurations.
                 *   The formula for the DLS solution is: delta_theta = (J^T * J + lambda^2 * I)^(-1) * J^T * Vb
                 *   where lambda is a damping factor that can be tuned based on the specific robot and application
                 */
                Eigen::MatrixXd Jt = Jb.transpose();
                int nJ = Jb.cols(); // number of joints

                // Compute the damped pseudo-inverse of the Jacobian
                Eigen::MatrixXd JJt_damped = Jt * Jb + (lambda * lambda * Eigen::MatrixXd::Identity(nJ, nJ));
                
                // Solver for the linear system (J^T * J + lambda^2 * I) * delta_theta = J^T * Vb
                Eigen::VectorXd dtheta = JJt_damped.ldlt().solve(Jt * Vb);
                
                thetalist += dtheta;
                Vb = Se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) * T));
                err = (Vb.head<3>().norm() > eomg) || (Vb.tail<3>().norm() > ev);

                maxiterations--;
            }

            return {thetalist, !err};
        }
        
        /** @brief Computes the inverse kinematics of a serial chain robot in the space frame using the Newton-Raphson method.  
         * * The logic and structure of this function is similar to the IKinBodyNR function, but it operates in the space frame {s} instead of the body frame {b}.
         * * The main differences are:
         * 1. The forward kinematics function used is FKinSpace instead of FKinBody, which computes the end-effector pose in the space frame.
         * 2. The Jacobian function used is JacobianSpace instead of JacobianBody, which computes the Space Jacobian
         * 3. The error twist is computed based on the difference between the current end-effector pose in the space frame and the desired pose T, using the same formula: Vs = Se3To
         * Vec(MatrixLog6(TransInv(T_current) * T)), where T_current is the current end-effector pose computed using FKinSpace.
         * 4. The rest of the algorithm proceeds in the same way, with iterative updates to the joint angles and recomputation of the error twist until convergence or maximum iterations.
         * @param T The desired end-effector pose in the space frame {s} as a 4x4 transformation matrix.
         * @param M The 4x4 home (all theta = 0) configuration matrix of the end-effector.
         * @param Slist The 6xn matrix of screw axes in the space frame {s} at the home position.
         * @param thetalist0 An initial guess for the joint angles, which can affect convergence.
         * @param eomg The error tolerance for the angular component of the end-effector twist (in radians).
         * @param ev The error tolerance for the linear component of the end-effector twist (in meters).
         * @return A pair consisting of:
         * - The computed joint angles as an Eigen::VectorXd.
         * - A boolean indicating whether the solution converged within the specified tolerance.
         */
        inline std::pair<Eigen::VectorXd, bool> IKinSpace(const Eigen::Matrix4d& T, const Eigen::Matrix4d& M,
        const Eigen::MatrixXd& Slist, const Eigen::VectorXd& thetalist0, double eomg, double ev) {
            // Similar implementation to IKinBody but using space frame calculations
            Eigen::VectorXd thetalist = thetalist0;
            int maxiterations = 20;
            Eigen::VectorXd Vs = Se3ToVec(MatrixLog6(T * TransInv(FKinSpace(M, Slist, thetalist))));
            bool err = (Vs.head<3>().norm() > eomg) || (Vs.tail<3>().norm() > ev);

            while (err && maxiterations > 0) {
                Eigen::MatrixXd Js = JacobianSpace(Slist, thetalist);
                
                if (isSingular(Js)) {
                    return {thetalist, false};
                }

                Eigen::VectorXd dtheta = Js.colPivHouseholderQr().solve(Vs);
                thetalist += dtheta;
                Vs = Se3ToVec(MatrixLog6(T * TransInv(FKinSpace(M, Slist, thetalist))));
                err = (Vs.head<3>().norm() > eomg) || (Vs.tail<3>().norm() > ev);

                maxiterations--;
            }
            return {thetalist, !err};
        }

        /** @brief Computes the inverse kinematics of a serial chain robot in the space frame using the Damped Least Squares (DLS) method.
         * * The logic and structure of this function is similar to the IKinBody function, but it operates in the space frame {s} instead of the body frame {b}.
         * * The main differences are:
         * 1. The forward kinematics function used is FKinSpace instead of FKinBody, which computes the end-effector pose in the space frame.
         * 2. The Jacobian function used is JacobianSpace instead of JacobianBody, which computes the Space Jacobian
         * 3. The error twist is computed based on the difference between the current end-effector pose in the space frame and the desired pose T, using the same formula: Vs = Se
         * ToVec(MatrixLog6(TransInv(T_current) * T)), where T_current is the current end-effector pose computed using FKinSpace.
         * 4. The change in joint angles (delta_theta) is computed using the Damped Least Squares method, which adds a damping term to the pseudo-inverse calculation to improve stability,
         * especially near singular configurations. The formula for the DLS solution is: delta_theta = (J^T * J + lambda^2 * I)^(-1) * J^T * Vs, where lambda is a damping factor that can be tuned based on the specific robot and application.
         * 5. The rest of the algorithm proceeds in the same way, with iterative updates to the joint angles and recomputation of the error twist until convergence or maximum iterations.
         * @param T The desired end-effector pose in the space frame {s} as a 4x4 transformation matrix.
         * @param M The 4x4 home (all theta = 0) configuration matrix of the end-effector.
         * @param Slist The 6xn matrix of screw axes in the space frame {s} at the home position.
         * @param thetalist0 An initial guess for the joint angles, which can affect convergence.
         * @param eomg The error tolerance for the angular component of the end-effector twist (in radians).
         * @param ev The error tolerance for the linear component of the end-effector twist (in meters).
         * @return A pair consisting of:
         * - The computed joint angles as an Eigen::VectorXd.
         * - A boolean indicating whether the solution converged within the specified tolerance. 
         */
        inline std::pair<Eigen::VectorXd, bool> IKinSpaceDLS(const Eigen::Matrix4d& T, const Eigen::Matrix4d& M,
        const Eigen::MatrixXd& Slist, const Eigen::VectorXd& thetalist0, double eomg, double ev) {
            // Similar implementation to IKinBodyDLS but using space frame calculations
            Eigen::VectorXd thetalist = thetalist0;
            int maxiterations = 50;
            double lambda = 0.1; // Damping factor
            Eigen::VectorXd Vs = Se3ToVec(MatrixLog6(T * TransInv(FKinSpace(M, Slist, thetalist))));
            bool err = (Vs.head<3>().norm() > eomg) || (Vs.tail<3>().norm() > ev);

            while (err && maxiterations > 0) {
                Eigen::MatrixXd Js = JacobianSpace(Slist, thetalist);
                
                // Check for singularity before computing the DLS solution
                if (isSingular(Js)) {
                    return {thetalist, false};
                }

                Eigen::MatrixXd Jt = Js.transpose();
                int nJ = Js.cols();

                // Compute the damped pseudo-inverse of the Jacobian
                Eigen::MatrixXd JJt_damped = Jt * Js + (lambda * lambda * Eigen::MatrixXd::Identity(nJ, nJ));
                Eigen::VectorXd dtheta = JJt_damped.ldlt().solve(Jt * Vs);
                
                thetalist += dtheta;
                Vs = Se3ToVec(MatrixLog6(T * TransInv(FKinSpace(M, Slist, thetalist))));
                err = (Vs.head<3>().norm() > eomg) || (Vs.tail<3>().norm() > ev);

                maxiterations--;
            }
            return {thetalist, !err};
        }
    }
}

#endif