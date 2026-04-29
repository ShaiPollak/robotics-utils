#ifndef MR_QUATERNIONS_H
#define MR_QUATERNIONS_H

#include "mr_core.h"
#include <Eigen/Dense>
#include <cmath>

/*
 * Quaternion math for the ModernRobotics library.
 *
 * Convention: unit quaternion q = (w, x, y, z) representing rotation,
 * where w is the scalar part and (x, y, z) is the vector part.
 *
 * Hamilton product: p * q  means p on the LEFT, q on the RIGHT.
 *   operator*(rhs) computes  this ⊗ rhs.
 *
 * Rotation matrix R = toRotationMatrix() maps a vector from body frame
 * to world frame: v_world = R * v_body.
 *
 * Euler angles follow the XYZ (Roll-Pitch-Yaw) convention in the fixed
 * (world) frame, matching the Python rotations.py reference implementation.
 */

namespace mr {

/** @brief Wraps an angle into the half-open interval (-π, π].
 *  @param a  Angle in radians.
 *  @return   Equivalent angle in the range (-π, π].
 */
inline double angleNormalize(double a) {
    a = std::fmod(a, 2.0 * M_PI);
    if (a <= -M_PI) a += 2.0 * M_PI;
    if (a >   M_PI) a -= 2.0 * M_PI;
    return a;
}

/** @brief Jacobian of XYZ Euler angles with respect to the axis-angle vector.
 *  Used for first-order covariance propagation from axis-angle space
 *  to roll-pitch-yaw space.
 *
 *  @param a  3D axis-angle vector (axis * angle), must be non-zero.
 *  @return   3×3 Jacobian J such that d(rpy) ≈ J * d(a).
 */
inline Eigen::Matrix3d rpyJacobianAxisAngle(const Eigen::Vector3d& a) {
    double na  = a.norm();
    double na3 = na * na * na;
    Eigen::Vector3d u = a / na;

    // Jr  (3×4): Jacobian of RPY w.r.t. (u, θ) where a = u*θ
    Eigen::Matrix<double, 3, 4> Jr = Eigen::Matrix<double, 3, 4>::Zero();
    Jr(0, 0) = na / (na * na * u(0) * u(0) + 1.0);
    Jr(0, 3) = u(0) / (na * na * u(0) * u(0) + 1.0);
    Jr(1, 1) = na / std::sqrt(1.0 - na * na * u(1) * u(1));
    Jr(1, 3) = u(1) / std::sqrt(1.0 - na * na * u(1) * u(1));
    Jr(2, 2) = na / (na * na * u(2) * u(2) + 1.0);
    Jr(2, 3) = u(2) / (na * na * u(2) * u(2) + 1.0);

    // Ja  (4×3): Jacobian of (u, θ) w.r.t. a
    Eigen::Matrix<double, 4, 3> Ja;
    // Top 3 rows: d(u)/d(a) = (θ²I - a*aᵀ) / θ³
    Ja.block<3, 3>(0, 0) = (na * na * Eigen::Matrix3d::Identity() - a * a.transpose()) / na3;
    // Bottom row: d(θ)/d(a) = a^T / θ = uᵀ
    Ja.row(3) = u.transpose();

    return Jr * Ja;
}


struct Quaternion {
    double w, x, y, z;

    // Identity quaternion by default.
    Quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) {}
    Quaternion(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}

    // ── Factory constructors ───────────────────────────────────────────────

    /** @brief Constructs a quaternion from a 4D vector (w, x, y, z).
     * @param v  4D vector with components (w, x, y, z).
     * @return   Quaternion with the given components.
     * Note: the input vector is not normalized, so the resulting quaternion may not be a unit quaternion.
     */
    static Quaternion fromVector(const Eigen::Vector4d& v) {
        return Quaternion(v(0), v(1), v(2), v(3));
    }

    /** @brief Constructs a quaternion from an axis-angle vector.
     * @param a  3D axis-angle vector, where the direction is the rotation
     *           axis and the magnitude is the rotation angle in radians.
     * @return   Quaternion representing the same rotation. For a zero vector, 
     *           returns the identity quaternion.
     * Note: the input vector is not normalized, so the resulting quaternion may not be a unit quaternion.
     */
    static Quaternion fromAxisAngle(const Eigen::Vector3d& a) {
        double norm = a.norm();
        Quaternion q;
        q.w = std::cos(norm / 2.0);
        if (norm < 1e-50) {
            q.x = 0.0; q.y = 0.0; q.z = 0.0;
        } else {
            Eigen::Vector3d imag = a / norm * std::sin(norm / 2.0);
            q.x = imag(0); q.y = imag(1); q.z = imag(2);
        }
        return q;
    }

    /** @brief Constructs a quaternion from XYZ Euler angles (roll, pitch, yaw).
     * @param rpy  3D vector of XYZ Euler angles (roll, pitch, yaw) in radians.
     * @return     Quaternion representing the same rotation.
     * Note: the input vector is not normalized, so the resulting quaternion may not be a unit quaternion.
     */
    static Quaternion fromEuler(const Eigen::Vector3d& rpy) {
        double cy = std::cos(rpy(2) * 0.5), sy = std::sin(rpy(2) * 0.5);
        double cp = std::cos(rpy(1) * 0.5), sp = std::sin(rpy(1) * 0.5);
        double cr = std::cos(rpy(0) * 0.5), sr = std::sin(rpy(0) * 0.5);
        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }

    // ── Conversions ────────────────────────────────────────────────────────

    /** @brief Converts the quaternion to a 4D vector (w, x, y, z).
     * @return 4D vector with components (w, x, y, z).
     */
    Eigen::Vector4d toVector() const { return Eigen::Vector4d(w, x, y, z); }

    /** @brief Converts the quaternion to an axis-angle vector.
     * @return 3D axis-angle vector, where the direction is the rotation axis and the magnitude is the rotation angle in radians.
     * Note: returns a zero vector for the identity quaternion.
     */
    Eigen::Vector3d toAxisAngle() const {
        double w_c = std::max(-1.0, std::min(1.0, w));
        double t   = 2.0 * std::acos(w_c);
        double s   = std::sin(t / 2.0);
        if (std::abs(s) < 1e-10) return Eigen::Vector3d::Zero();
        return (t / s) * Eigen::Vector3d(x, y, z);
    }

    /** @brief Converts the quaternion to a rotation matrix.
     * @return 3x3 rotation matrix R such that v_world = R * v_body.
     */
    Eigen::Matrix3d toRotationMatrix() const {
        Eigen::Vector3d v(x, y, z);
        return (w * w - v.dot(v)) * Eigen::Matrix3d::Identity()
               + 2.0 * v * v.transpose()
               + 2.0 * w * VecToSo3(v);
    }

    /** @brief Converts the quaternion to XYZ Euler angles (roll, pitch, yaw).
     * @return 3D vector of XYZ Euler angles (roll, pitch, yaw) in radians.
     */
    Eigen::Vector3d toEuler() const {
        double roll  = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
        double pitch = std::asin(std::max(-1.0, std::min(1.0, 2.0 * (w * y - z * x))));
        double yaw   = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
        return Eigen::Vector3d(roll, pitch, yaw);
    }

    // ── Arithmetic ─────────────────────────────────────────────────────────

    /** @brief Hamilton product: this ⊗ rhs
     * @param rhs  Right-hand side quaternion.
     * @return     Resulting quaternion from the Hamilton product.
     * Note: the result is not normalized, so it may not be a unit quaternion even if both operands are unit quaternions.
     */
    Quaternion operator*(const Quaternion& rhs) const {
        return Quaternion(
            w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
            w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
            w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w
        );
    }

    /** @brief Conjugate of the quaternion.
     * @return Conjugate quaternion q* = (w, -x, -y, -z). For unit quaternions, equals inverse.
     */
    Quaternion conjugate() const { return Quaternion(w, -x, -y, -z); }

    /** @brief Inverse of the quaternion.
     * @return Inverse quaternion. For unit quaternions, equals conjugate.
     */
    Quaternion inverse() const {
        double n2 = w*w + x*x + y*y + z*z;
        return Quaternion(w / n2, -x / n2, -y / n2, -z / n2);
    }

    /** @brief Normalizes the quaternion to unit length.
     * @return Quaternion with unit norm.
     */
    Quaternion normalize() const {
        double n = std::sqrt(w*w + x*x + y*y + z*z);
        return Quaternion(w / n, x / n, y / n, z / n);
    }

    // ── Left / right multiplication matrices ──────────────────────────────
    // Useful for Jacobian derivations.

    /** @brief Left multiplication matrix M_L such that 
     *         M_L(this) * q.toVector() == (this * q).toVector()
     * @return 4x4 matrix M_L such that M_L(this) * q.toVector() == (this * q).toVector() 
     *         for any quaternion q.
     * Note: the resulting matrix is not orthogonal, and does not preserve norms, 
     * since quaternion multiplication is not a linear operation in the 4D vector space.
     */
    Eigen::Matrix4d leftMatrix() const {
        Eigen::Matrix4d M;
        M <<  w, -x, -y, -z,
              x,  w, -z,  y,
              y,  z,  w, -x,
              z, -y,  x,  w;
        return M;
    }

    /** @brief Right multiplication matrix M_R such that 
     *         M_R(this) * q.toVector() == (q * this).toVector()
     * @return 4x4 matrix M_R such that M_R(this) * q.toVector() == (q * this).toVector() 
     *         for any quaternion q.
     * Note: the resulting matrix is not orthogonal, and does not preserve norms, 
     * since quaternion multiplication is not a linear operation in the 4D vector space.
     */
    Eigen::Matrix4d rightMatrix() const {
        Eigen::Matrix4d M;
        M <<  w, -x, -y, -z,
              x,  w,  z, -y,
              y, -z,  w,  x,
              z,  y, -x,  w;
        return M;
    }

};

} // namespace mr

#endif // MR_QUATERNIONS_H
