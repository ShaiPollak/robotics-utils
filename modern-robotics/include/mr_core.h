#ifndef MR_CORE_H
#define MR_CORE_H

#include <Eigen/Dense>
#include <cmath>

/* Core mathematical functions (lie algebras) for the ModernRobotics library:
- Matrix logarithm and exponential for SE(3) and SO(3)
- Adjoint transformations
- Screw axis representations
- Trajectory generation utilities
*/

namespace mr {

    /** @brief Threshold for determining if a value is close enough to zero to be considered zero.
     * This is used to handle numerical stability issues when dealing with small angles or near-singular configurations. */
    const double nearly_zero_threshold = 1e-8;

    inline bool nearZero(double val) {
        return std::abs(val) < nearly_zero_threshold;
    }

    // COVERSIONS BETWEEN VECTOR AND MATRIX REPRESENTATIONS OF LIE ALGEBRAS

    /** @brief Converts a 3D angular velocity vector into a 3x3 skew-symmetric matrix (so(3) representation). */
    inline Eigen::Matrix3d VecToSo3(const Eigen::Vector3d& omega) {
        Eigen::Matrix3d so3mat;
        so3mat <<     0, -omega(2),  omega(1),
                   omega(2),      0, -omega(0),
                  -omega(1),  omega(0),      0;
        return so3mat;
    }

    /** @brief Converts a 3x3 skew-symmetric matrix (so(3) representation) back into a 3D angular velocity vector.
     */
    inline Eigen::Vector3d So3ToVec(const Eigen::Matrix3d& so3mat) {
        Eigen::Vector3d omega;
        omega << so3mat(2, 1), so3mat(0, 2), so3mat(1, 0);
        return omega;
    }

        /**
     * @brief Converts a 6D twist vector into a 4x4 se(3) matrix.
     * * Takes a 6D vector V = [w, v] and maps it to the se(3) Lie Algebra representation:
     * [ [w]  v ]
     * [  0   0 ]
     * * @param V A 6D vector (top 3: angular velocity w, bottom 3: linear velocity v).
     * @return 4x4 se(3) matrix.
     * * @usage_future
     * 1. Matrix Exponential (MatrixExp6): Essential for converting velocities to 
     * actual transformations (T).
     * 2. Forward Kinematics (FKin): Used in the Product of Exponentials (PoE) formula 
     * to represent joint axes.
     * 3. Numerical Inverse Kinematics: Used to update the robot's configuration 
     * based on calculated twist errors.
     */
    inline Eigen::Matrix4d VecToSe3(const Eigen::VectorXd& V) {
        Eigen::Matrix4d se3mat = Eigen::Matrix4d::Zero();
        se3mat.block<3, 3>(0, 0) = VecToSo3(V.head(3));
        se3mat.block<3, 1>(0, 3) = V.tail(3);
        return se3mat;
    }

    /**
     * @brief Converts a 4x4 se(3) matrix back into a 6D twist vector.
     * * Takes a 4x4 skew-symmetric matrix and returns the corresponding 6D vector:
     * [w]
     * [v]
     * * @param se3mat A 4x4 se(3) matrix.
     * @return 6D twist vector.
     * * @usage_future
     * 1. Matrix Logarithm (MatrixLog6): Essential for converting transformations back
     * to velocities, especially when calculating errors in inverse kinematics.
     * 2. Feedback Control: Used to compute the twist error between the current
     * end-effector pose and the desired pose, which is then used to generate control
     */
    inline Eigen::Matrix<double, 6, 1> Se3ToVec(const Eigen::Matrix4d& se3mat) {
        Eigen::Matrix<double, 6, 1> V; 
        V.head(3) = So3ToVec(se3mat.block<3, 3>(0, 0));
        V.tail(3) = se3mat.block<3, 1>(0, 3);
        return V;
    }


    // TRANSFORMATIONS

    /** @brief Converts a rotation matrix R and a position vector p into a homogeneous transformation matrix T. */
    inline Eigen::Matrix4d RpToTrans(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = p;
        return T;
    }

    /** @brief Converts a homogeneous transformation matrix T into a rotation matrix R and a position vector p. */
    inline std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransToRp(const Eigen::Matrix4d& T) {
        Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        Eigen::Vector3d p = T.block<3, 1>(0, 3);
        return std::make_pair(R, p);
    }

    /** @brief Computes the inverse of a homogeneous transformation matrix T.
     * The inverse of a transformation T = [R, p; 0, 1] is given by:
     * T_inv = [R^T, -R^T * p; 0, 1]
     * This function is essential for changing reference frames and is used in various
     * kinematic calculations, such as computing the relative transformation between two
    * poses or transforming twists and wrenches between different frames.
    */
    inline Eigen::Matrix4d TransInv(const Eigen::Matrix4d& T) {
        Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        Eigen::Vector3d p = T.block<3, 1>(0, 3);
        Eigen::Matrix4d T_inv = Eigen::Matrix4d::Identity();
        T_inv.block<3, 3>(0, 0) = R.transpose();
        T_inv.block<3, 1>(0, 3) = -R.transpose() * p;
        return T_inv;
    }

    /**
     * @brief Computes the 6x6 Adjoint representation of a homogeneous transformation matrix T.
     * * The Adjoint matrix [AdT] is used to transform twists (6D velocities) from one 
     * frame to another. For a given T = [R, p; 0, 1], the Adjoint is:
     * [  R     0  ]
     * [ [p]R   R  ]
     * * @param T A 4x4 homogeneous transformation matrix (SE(3)).
     * @return A 6x6 Adjoint transformation matrix.
     * * @usage_future
     * 1. Frame Transformation: Mapping a twist from the body frame {b} to the 
     * space frame {s}: Vs = [Ad_Tsb] * Vb.
     * 2. Jacobian Change of Frames: Converting a Body Jacobian to a Space Jacobian.
     * 3. Dynamics: Transforming wrenches (forces/torques) between different 
     * points on a robot link.
     */
    inline Eigen::Matrix<double, 6, 6> Adjoint(const Eigen::Matrix4d& T) {
        auto [R, p] = TransToRp(T);
        Eigen::Matrix<double, 6, 6> AdT = Eigen::Matrix<double, 6, 6>::Zero();
        AdT.block<3, 3>(0, 0) = R;
        AdT.block<3, 3>(3, 0) = VecToSo3(p) * R;
        AdT.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
        AdT.block<3, 3>(3, 3) = R;
        return AdT; 
    }

    // Matrix Exponential and Logarithm functions for SE(3) and SO(3) are defined in trajectory.h since they are primarily used for trajectory generation and control.

    /**
     * @brief Converts a 3D exponential coordinate vector to axis-angle representation.
     * * Formula: theta = ||expc3||, omghat = expc3 / theta.
     * * @param expc3 The 3D exponential coordinate vector (omg * theta).
     * @return A std::pair with the unit axis (omghat) and the rotation angle (theta).
     * * @usage_future
     * 1. MatrixExp3: To handle cases where no rotation is needed (theta approx 0).
     * 2. Singularity Handling: Prevents division by zero in the motion planning pipeline.
     * * @note If theta is near zero, omghat is undefined. The function returns a zero vector.
     */
    inline std::pair<Eigen::Vector3d, double> AxisAng3(Eigen::Vector3d& expc3) {
        double theta = expc3.norm();
        Eigen::Vector3d omghat;
        if (nearZero(theta)) {
            omghat = Eigen::Vector3d::Zero();
        } else {
            omghat = expc3 / theta; // Normalize to get the rotation axis
        }
        return std::make_pair(omghat, theta);
    
    }
    
    /**
     * @brief Decomposes a 6D exponential coordinate vector into a screw axis and a travel distance.
     * * Formula:
     * 1. If rotation exists: theta = ||omg*theta||, S = expc6 / theta.
     * 2. If pure translation: theta = ||v*theta||, S = expc6 / theta (where S is [0; v_unit]).
     * * @param expc6 A 6D vector of exponential coordinates for a rigid-body motion (S*theta).
     * @return A std::pair containing:
     * - first: Eigen::Matrix<double, 6, 1> S (The normalized screw axis).
     * - second: double theta (The distance: angle for rotation, or linear distance for translation).
     * * @usage_future
     * 1. MatrixLog6: AxisAng6 is the final step to extract the physical parameters of motion.
     * 2. Motion Analysis: Understanding if a robot's move is more "rotational" or "translational".
     */
    inline std::pair<Eigen::Matrix<double, 6, 1>, double> AxisAng6(const Eigen::Matrix<double, 6, 1>& expc6) {
    double theta = expc6.head(3).norm();
    Eigen::Matrix<double, 6, 1> S;

    if (nearZero(theta)) { 
        // Pure translation case: theta is the norm of the linear part, and the screw axis is [0; v_unit]
        theta = expc6.tail(3).norm();
        if (nearZero(theta)) {
            // No motion, return zero screw axis and zero theta
            S.setZero();
        } else {
            S = expc6 / theta; // כאן החלק העליון יצא אפס ממילא
        }
    } else {
        // Rotation exists
        S = expc6 / theta;
    }

    return std::make_pair(S, theta);
}

    
    /**
     * @brief Generates a normalized screw axis representation (S).
     * * Combines the geometric description of a screw (point, direction, pitch) 
     * into a single 6D vector S = [omega; v].
     * * Formula:
     * S = [ s; (q x s) + h*s ]
     * * @param q A point on the screw axis (3D vector).
     * @param s A unit vector in the direction of the screw axis (3D vector).
     * @param h The pitch of the screw axis (scalar) - zero for a pure rotation.
     * @return A 6D normalized screw axis (Vector6d).
     * * @usage_future
     * 1. Product of Exponentials (PoE): This S is what you feed into MatrixExp6 
     * to calculate the robot's forward kinematics.
     * 2. Jacobian: Each column of a Space Jacobian is a screw axis representing a joint.
     * 3. Defining Joints: For the youBot, each wheel and arm joint will be 
     * represented by such an S vector.
     */
    inline Eigen::Matrix<double, 6, 1> ScrewToAxis(const Eigen::Vector3d& q, const Eigen::Vector3d& s, double h) {
        Eigen::Matrix<double, 6, 1> S;
        S.head(3) = s;
        S.tail(3) = q.cross(s) + h * s;
        return S;
    }

    /**
     * @brief Computes the matrix exponential of a 3D exponential coordinate vector (so(3) representation).
     * * Uses Rodrigues' formula: R = I + sin(theta)*[omghat] + (1-cos(theta))*[omghat]^2.
     * * @param so3mat A 3x3 skew-symmetric matrix representing the exponential coordinates (omg * theta).
     * @return A 3x3 rotation matrix corresponding to the exponential coordinates.
     * * @usage_future
     * 1. Trajectory Generation: Converting angular velocity commands into actual rotations.
     * 2. Forward Kinematics: Used in the Product of Exponentials formula to compute the end-effector pose from joint angles.
     * 3. Feedback Control: Converting twist errors into corrective transformations for the end-effector.
     * * @note If the input represents a zero rotation (theta approx 0), the function returns the identity matrix.
     */
    inline Eigen::Matrix3d MatrixExp3(const Eigen::Matrix3d& so3mat) {
        Eigen::Vector3d omgtheta = So3ToVec(so3mat);
        if (nearZero(omgtheta.norm())) {
            return Eigen::Matrix3d::Identity(); // No rotation
        } else {
            auto [omghat, theta] = AxisAng3(omgtheta);
            Eigen::Matrix3d omghatmat = VecToSo3(omghat);
            return Eigen::Matrix3d::Identity() + sin(theta) * omghatmat + (1 - cos(theta)) * omghatmat * omghatmat;
        }
    }
    
    /**
     * @brief Computes the matrix logarithm of a rotation matrix R.
     * * This extracts the exponential coordinates [omg]*theta from SO(3) to so(3).
     * Formula: theta = acos((tr(R)-1)/2), [omg] = (R - R^T) / (2*sin(theta)).
     * * @param R A 3x3 rotation matrix.
     * @return A 3x3 skew-symmetric matrix (so(3)).
     * * @usage_future
     * 1. Orientation Error: Calculating the difference between desired and current pose.
     * 2. Screw Trajectory: Determining the rotational part of the screw motion.
     */
    inline Eigen::Matrix3d MatrixLog3(const Eigen::Matrix3d& R) {
        // 1. Clamping the input to avoid NaN from floating point errors
        double acosinput = std::max(-1.0, std::min(1.0, (R.trace() - 1.0) / 2.0));
        
        if (acosinput >= 1.0) {
            return Eigen::Matrix3d::Zero(); // No rotation (theta = 0)
        } 
        else if (acosinput <= -1.0) {
            // Case: 180 degree rotation (theta = pi). R is symmetric.
            Eigen::Vector3d omg;
            if (1.0 + R(2, 2) > nearly_zero_threshold) {
                omg = (1.0 / std::sqrt(2.0 * (1.0 + R(2, 2)))) * Eigen::Vector3d(R(0, 2), R(1, 2), 1.0 + R(2, 2));
            } else if (1.0 + R(1, 1) > nearly_zero_threshold) {
                omg = (1.0 / std::sqrt(2.0 * (1.0 + R(1, 1)))) * Eigen::Vector3d(R(0, 1), 1.0 + R(1, 1), R(2, 1));
            } else {
                omg = (1.0 / std::sqrt(2.0 * (1.0 + R(0, 0)))) * Eigen::Vector3d(1.0 + R(0, 0), R(1, 0), R(2, 0));
            }
            return VecToSo3(M_PI * omg);
        } 
        else {
            // General case
            double theta = std::acos(acosinput);
            return (theta / (2.0 * std::sin(theta))) * (R - R.transpose());
        }
    }

    /**
     * @brief Computes the 4x4 homogeneous transformation matrix SE(3) from se(3).
     * * This represents traveling along a screw axis S for a distance theta.
     * Formula: T = [ Exp([omg]*theta), G(theta)*v ]
     * [        0        ,      1       ]
     * where G(theta) = I*theta + (1-cos(theta))*[omg] + (theta-sin(theta))*[omg]^2
     * * @param se3mat A 4x4 matrix in the Lie Algebra se(3).
     * @return A 4x4 transformation matrix in SE(3).
     * * @usage_future
     * 1. Forward Kinematics: The PoE formula is T = Exp(S1*th1)*Exp(S2*th2)...*M.
     * 2. Trajectory Generation: Converting intermediate twist steps into poses.
     */
    inline Eigen::Matrix4d MatrixExp6(const Eigen::Matrix4d& se3mat) {
        Eigen::Matrix3d so3mat = se3mat.block<3, 3>(0, 0);
        Eigen::Vector3d omgtheta = So3ToVec(so3mat);
        double theta = omgtheta.norm();

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

        if (nearZero(theta)) {
            // Pure Translation: T = [I, v; 0, 1]
            T.block<3, 1>(0, 3) = se3mat.block<3, 1>(0, 3);
        } else {
            // Combined case: Using the G matrix
            Eigen::Matrix3d omgmat = so3mat / theta; // This is the [omghat] from the book
            
            Eigen::Matrix3d G = Eigen::Matrix3d::Identity() * theta + 
                                (1.0 - std::cos(theta)) * omgmat + 
                                (theta - std::sin(theta)) * (omgmat * omgmat);

            T.block<3, 3>(0, 0) = MatrixExp3(so3mat);
            T.block<3, 1>(0, 3) = G * (se3mat.block<3, 1>(0, 3) / theta);
        }
        
        return T;
    }

    /**
     * @brief Computes the matrix logarithm of an SE(3) matrix.
     * * Maps SE(3) to se(3). This is the inverse of MatrixExp6.
     * It finds the twist (omg, v) that produces the transformation T.
     * * @param T A 4x4 homogeneous transformation matrix.
     * @return A 4x4 matrix in the Lie Algebra se(3).
     * * @usage_future
     * 1. Screw Trajectory: To find the "delta" motion between two poses.
     * 2. Feedback Control: To compute the 6D error twist between current and desired T.
     */
    inline Eigen::Matrix4d MatrixLog6(const Eigen::Matrix4d& T) {
        auto [R, p] = TransToRp(T);
        Eigen::Matrix3d omgmat = MatrixLog3(R);
        Eigen::Matrix4d se3mat = Eigen::Matrix4d::Zero();

        double theta = So3ToVec(omgmat).norm();

        if (nearZero(theta)) {
            // No Rotation, Pure Translation: se3mat = [0, v; 0, 0]
            se3mat.block<3, 3>(0, 0) = Eigen::Matrix3d::Zero();
            se3mat.block<3, 1>(0, 3) = p;
        } else {
            // Combined case: Computing the linear velocity through G_inv
            Eigen::Matrix3d omgmat_norm = omgmat / theta;
            
            Eigen::Matrix3d G_inv = Eigen::Matrix3d::Identity() / theta - 
                                    0.5 * omgmat_norm + 
                                    (1.0 / theta - 0.5 / std::tan(theta / 2.0)) * (omgmat_norm * omgmat_norm);

            se3mat.block<3, 3>(0, 0) = omgmat;
            se3mat.block<3, 1>(0, 3) = G_inv * (p * theta); 
        }
        
        return se3mat;
    }

    /** @brief Computes the 6x6 adjoint representation of a twist V.
     * * The adjoint of a twist is used to compute the effect of one twist on another, especially in the context of the Lie bracket [V1, V2] = ad_V
     * * Formula: ad_V = [ [w], [v]; 0, [w] ]
     * where V = [w; v] is the twist. The top-left and bottom-right blocks are the skew-symmetric matrix of w, and the top-right block is the skew-symmetric matrix of v.
     * * @param V A 6D twist vector (angular velocity w and linear velocity v).
     * @return A 6x6 matrix representing the adjoint of the twist V
     * * @usage_future
     * 1. Lie Bracket: The Lie bracket of two twists V1 and V2 can be computed as [V1, V2] = ad_V1 * V2.
     * 2. Dynamics: In the equations of motion, the Coriolis and centrifugal terms can be expressed using the adjoint representation of twists.
     * 3. Control: When designing controllers that involve the interaction of multiple twists 
     * (e.g., in a multi-robot system), the adjoint can be used to compute how one robot's motion affects another's.
     */
    inline Eigen::Matrix<double, 6, 6> adVec(const Eigen::Matrix<double, 6, 1>& V) {
        Eigen::Matrix<double, 6, 6> adV = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Vector3d omega = V.head(3);
        Eigen::Vector3d v = V.tail(3);

        adV.block<3, 3>(0, 0) = VecToSo3(omega);
        adV.block<3, 3>(0, 3) = VecToSo3(v);
        adV.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
        adV.block<3, 3>(3, 3) = VecToSo3(omega);

        return adV;
    }

    /** @brief Computes the distance from a 3x3 matrix to the nearest rotation matrix in the Frobenius norm.
     * Formula: distance = ||R^T * R - I|| + |det(R) - 1|
     * * @param R A 3x3 matrix.
     * @return The distance from R to the nearest rotation matrix.
     */
    inline double distanceToSO3(const Eigen::Matrix3d& R) {
        if (R.determinant() < 0) {
            return std::numeric_limits<double>::infinity(); // Not a valid rotation matrix
        }
        return (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() + std::abs(R.determinant() - 1.0);

    }
    
    /** @brief Computes the distance from a 4x4 matrix to the nearest SE(3) transformation in the Frobenius norm.
     * Formula: distance = distanceToSO3(R) + ||p||, where R is the top-left 3x3 block and p is the top-right 3x1 block
     * * @param T A 4x4 matrix.
     * @return The distance from T to the nearest SE(3) transformation.
     */
    inline double distanceToSE3(const Eigen::Matrix4d& T) {
        auto [R, p] = TransToRp(T);
        if (R.determinant() < 0) {
            return std::numeric_limits<double>::infinity(); // Not a valid rotation matrix
        }
        double rot_distance = (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() + std::abs(R.determinant() - 1.0); 
        double trans_distance = p.norm();
        return rot_distance + trans_distance;
    }

    /** @brief Tests if a 3x3 matrix is a valid rotation matrix.
     * * @param mat A 3x3 matrix.
     * @return True if the matrix is a valid rotation matrix, false otherwise.
     */
    inline bool testIfSo3(const Eigen::Matrix3d& mat) {
        return distanceToSO3(mat) < nearly_zero_threshold;
    }

    /** @brief Tests if a 4x4 matrix is a valid SE(3) transformation.
     * * @param mat A 4x4 matrix.
     * @return True if the matrix is a valid SE(3) transformation, false otherwise.
     */
    inline bool testIfSe3(const Eigen::Matrix4d& mat) {
        return testIfSo3(mat.block<3, 3>(0, 0)) && distanceToSE3(mat) < nearly_zero_threshold;
    }

    /** @brief Computes the pseudo-inverse of a matrix using singular value decomposition.
     * * @param mat The matrix for which to compute the pseudo-inverse.
     * @return The pseudo-inverse of the matrix.
     */
    inline Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& mat) {
        // 1. Use JacobiSVD with ThinU and ThinV
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& singularValues = svd.singularValues();
        
        // 2. Create a square diagonal matrix based on the number of singular values
        // For a 6x9 matrix, this will be 6x6.
        int n = singularValues.size();
        Eigen::MatrixXd singularValuesInv = Eigen::MatrixXd::Zero(n, n);

        for (int i = 0; i < n; ++i) {
            if (singularValues(i) > nearly_zero_threshold) {
                singularValuesInv(i, i) = 1.0 / singularValues(i);
            }
        }

        // 3. Perform the multiplication:
        return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
    }

    /** @brief Computes the damped pseudo-inverse of a matrix using singular value decomposition.
     * * This is used to handle near-singular configurations in inverse kinematics by adding a damping term to the singular values.
     * * We define a cost function: L = ||A * x - b||^2 + lambda^2 * ||x||^2, where lambda is the damping factor.
     * * Then we take the derivative with respect to x and set it to zero:
     * * (J^T * J + lambda^2 * I) * x = J^T * b
     * * Then x = (J^T * J + lambda^2 * I)^-1 * J^T * b, which is the damped pseudo-inverse solution.
     * * We return only the damped pseudo-inverse matrix: J^T * (J * J^T + lambda^2 * I)^-1 (which is more efficient to compute).
     * * @param A The matrix for which to compute the damped pseudo-inverse.
     * @param lambda The damping factor (default is 0.1). Higher values increase stability but reduce accuracy near singularities.
     * @return The damped pseudo-inverse of the matrix.
     */
    inline Eigen::MatrixXd dampedPseudoInverse(const Eigen::MatrixXd& A, double lambda=0.1) {
        // Formula: J_damped_inv = J^T * (J * J^T + lambda^2 * I)^-1
        Eigen::MatrixXd AAT = A * A.transpose();
        Eigen::MatrixXd damping = (lambda * lambda) * Eigen::MatrixXd::Identity(A.rows(), A.rows());
        
        // Use LDLT for numerical stability and speed
        return A.transpose() * (AAT + damping).ldlt().solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
    }

    

    /** @brief Tests if a transformation matrix has no rotation.
     * * @param T A 4x4 transformation matrix.
     * @return True if the matrix has no rotation, false otherwise.
     */
    inline bool noRotation(const Eigen::Matrix4d& T) {
        auto [R, p] = TransToRp(T);
        return testIfSo3(R) && R.isApprox(Eigen::Matrix3d::Identity(), nearly_zero_threshold);
    }

    
    /** @brief Tests if a rotation matrix has no rotation.
     * * @param R A 3x3 rotation matrix.
     * @return True if the matrix has no rotation, false otherwise.
     */
    inline bool noRotationSO3(const Eigen::Matrix3d& R) {
        return testIfSo3(R) && R.isApprox(Eigen::Matrix3d::Identity(), nearly_zero_threshold);
    }

    /** @brief Tests if a twist vector has no rotational component.
     * * @param V A 6D twist vector.
     * @return True if the twist has no rotational component, false otherwise.
     */
    inline bool noRotationSE3(const Eigen::Matrix4d& T) {
        auto [R, p] = TransToRp(T);
        return noRotationSO3(R);
    }


    /** @brief Tests if a twist vector has no rotational component.
     * * @param V A 6D twist vector.
     * @return True if the twist has no rotational component, false otherwise.
     */
    inline bool noRotation(const Eigen::VectorXd& V) {
        return V.head(3).norm() < nearly_zero_threshold;
    }

    /** @brief Normalizes an angle to be within the range [-pi, pi].
     * * This is useful for ensuring that angle representations remain consistent and do not exceed the standard
     *  range for angles, which can help prevent issues with angle wrapping in control algorithms and trajectory generation.
     * * @param angle The angle to be normalized (in radians).
     * @return The normalized angle within the range [-pi, pi].
     */
    inline double normalizeAngle(double angle) {
        return atan2(sin(angle), cos(angle));
    }

    /** @brief Performs Euler integration to update a state based on its derivative and a time step.
     * * This is a simple numerical integration method that approximates the new state as:
     * new_state = initial_state + derivative * dt
     * It is commonly used in control algorithms to update the robot's configuration based on calculated velocities or accelerations.
     * * @param initial The initial state vector (e.g., joint angles, end-effector pose).
     * @param derivative The derivative of the state (e.g., joint velocities, twist).
     * @param dt The time step for integration.
     * @return The updated state vector after applying Euler integration.
     */
    inline Eigen::VectorXd eulerIntegration(const Eigen::VectorXd& initial, const Eigen::VectorXd& derivative, double dt) {
        return initial + derivative * dt;
    }
}

#endif