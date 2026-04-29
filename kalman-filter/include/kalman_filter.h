#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <Eigen/Dense>
#include "../../modern-robotics/include/mr_quaternions.h"

/*
 * Error-State Extended Kalman Filter (ES-EKF) for IMU-centric navigation.
 *
 * State (nominal, 10 DOF):
 *   p  — position     (3)
 *   v  — velocity     (3)
 *   q  — orientation  (4, unit quaternion wxyz)
 *
 * Error state (9 DOF), linearized around the nominal state:
 *   δp  — position error     (3)
 *   δv  — velocity error     (3)
 *   δθ  — orientation error  (3, axis-angle)
 *
 * Process input: IMU specific force  f  and angular rate  ω  (both body frame).
 *
 * ── Adding new sensors ────────────────────────────────────────────────────
 * For any sensor that measures a linear function of the error state:
 *   1. Compute the innovation:   ν = z_measured − h(nominal_state)
 *   2. Build observation Jacobian H  (m × 9) and noise covariance R  (m × m).
 *   3. Call  esekf.update(ν, H, R).
 *
 * Position-only sensors (GNSS, LiDAR-as-position) can use the convenience
 * wrapper  updatePosition()  which fills H = [I₃ | 0 | 0] automatically.
 *
 * ── Adding new filter variants ────────────────────────────────────────────
 * Inherit from  KalmanFilterBase<N>  and provide your own  predict() method.
 * The free function  ekfMeasurementUpdate()  is reusable by any linear
 * observer (EKF, IEKF, UKF sigma-point update, etc.).
 *
 * Dependencies: Eigen 3, mr_quaternions.h (mr_core.h, VecToSo3).
 */

namespace kf {

/** @brief Return type of ekfMeasurementUpdate.
  * @tparam N  Dimension of the error (or full) state vector.
  */
template<int N>
struct EKFUpdateResult {
    Eigen::Matrix<double, N, 1> dx;  // error-state correction to inject into nominal state
    double nis;                       // Normalized Innovation Squared: νᵀ·S⁻¹·ν ~ χ²(m)
                                      // Expected value = m (measurement dimension).
                                      // Large nis → measurement inconsistent with filter uncertainty.
};

/** @brief Performs the EKF measurement update step.
  * Relevant Formulas:
  *  S   = H·P·Hᵀ + R        (innovation covariance)
  *  K   = P·Hᵀ·S⁻¹           (Kalman gain)
  *  dx  = K·ν                (error-state correction)
  *  P⁺  = (I − K·H)·P        (covariance update)
  *  NIS = νᵀ·S⁻¹·ν  ~ χ²(m)  (Normalized Innovation Squared — measurement reliability)
  * @tparam N  Dimension of the error (or full) state vector.
  * @param P           Prior error covariance (N×N), updated in-place to the posterior.
  * @param innovation  Measurement innovation (z − h(x̂)) (m×1).
  * @param H           Observation Jacobian mapping error state to measurements (m×N).
  * @param R           Measurement noise covariance (m×m).
  * @return            EKFUpdateResult containing dx (correction) and nis (reliability score).
  * Note: the caller is responsible for applying dx to the nominal state.
 */
template<int N>
inline EKFUpdateResult<N> ekfMeasurementUpdate(
    Eigen::Matrix<double, N, N>& P,
    const Eigen::VectorXd&       innovation,
    const Eigen::MatrixXd&       H,
    const Eigen::MatrixXd&       R)
{
    // S = H·P·Hᵀ + R  (innovation covariance)
    const Eigen::MatrixXd S     = H * P * H.transpose() + R;
    const Eigen::MatrixXd S_inv = S.inverse();
    // Kalman gain K (N×m)
    const Eigen::MatrixXd K = P * H.transpose() * S_inv;
    // Normalized Innovation Squared — reuses S_inv already computed for K
    const double nis = (innovation.transpose() * S_inv * innovation).value();
    // Error-state correction
    const Eigen::Matrix<double, N, 1> dx = K * innovation;
    // Joseph-form covariance update (numerically symmetric)
    const Eigen::Matrix<double, N, N> IKH = Eigen::Matrix<double, N, N>::Identity() - K * H;
    P = IKH * P;
    return {dx, nis};
}


// ── NIS outlier detection ─────────────────────────────────────────────────

/** @brief Chi-squared threshold at a given confidence level.
 *
 *  Under a well-tuned filter, NIS ~ χ²(m).  A measurement is an outlier if
 *  its NIS exceeds this threshold.
 *
 *  Uses a pre-computed lookup table for m = 1…12; falls back to the
 *  Wilson-Hilferty cube-root approximation (< 1 % error for m ≥ 2) otherwise.
 *
 *  Supported confidence levels: 0.90, 0.95 (default), 0.99.
 *  Values outside these bands clamp to the nearest supported level.
 *
 *  @param m           Measurement dimension (degrees of freedom).
 *  @param confidence  Desired confidence level (0.90 / 0.95 / 0.99).
 *  @return            χ²(m) quantile at the requested confidence.
 */
inline double chi2Threshold(int m, double confidence = 0.95) {
    // Pre-computed quantiles: rows = [90%, 95%, 99%], cols = m = 1…12
    static const double kTable[3][12] = {
        // 90 %
        { 2.706, 4.605, 6.251, 7.779, 9.236, 10.645, 12.017, 13.362, 14.684, 15.987, 17.275, 18.549 },
        // 95 %
        { 3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026 },
        // 99 %
        { 6.635, 9.210, 11.345, 13.277, 15.086, 16.812, 18.475, 20.090, 21.666, 23.209, 24.725, 26.217 }
    };
    // Normal quantiles z_α for Wilson-Hilferty fallback (m > 12)
    static const double kZ[3] = { 1.2816, 1.6449, 2.3263 };

    int row = (confidence < 0.92) ? 0 : (confidence < 0.97) ? 1 : 2;

    if (m >= 1 && m <= 12)
        return kTable[row][m - 1];

    // Wilson-Hilferty approximation for large m
    double z = kZ[row];
    double h = 1.0 - 2.0 / (9.0 * m) + z * std::sqrt(2.0 / (9.0 * m));
    return m * h * h * h;
}

/** @brief Returns true if the NIS value indicates a potential measurement outlier.
 *
 *  Compares NIS against the χ²(m) threshold at the given confidence level.
 *  A return value of true means the measurement is statistically inconsistent
 *  with the filter's current uncertainty — the caller should decide whether to
 *  reject the measurement, re-tune the sensor variance, or investigate further.
 *
 *  Typical usage:
 *  @code
 *      double nis = filter.updatePosition(gnss_pos, var_gnss);
 *      if (kf::isOutlier(nis, 3))   // 3-DOF position sensor, 95% default
 *          disableGnss();
 *  @endcode
 *
 *  @param nis         Normalized Innovation Squared returned by any update call.
 *  @param m           Measurement dimension (3 for position, 6 for pose, etc.).
 *  @param confidence  Confidence level for the chi-squared test (default 0.95).
 *  @return            true if nis > χ²(m, confidence).
 */
inline bool isOutlier(double nis, int m, double confidence = 0.95) {
    return nis > chi2Threshold(m, confidence);
}


/** @brief Base class for all Kalman filter variants.
  * @tparam N  Dimension of the error (or full) state vector.
  * Provides a unified interface for state covariance access and generic
  * measurement updates.  Concrete filters override predict() and optionally
  * update() with sensor-specific convenience wrappers.
  */
template<int N>
class KalmanFilterBase {
public:
    virtual ~KalmanFilterBase() = default;

    /** @brief Generic measurement update.
      * @param innovation  z − h(x̂)  pre-computed by the caller.
      * @param H           Observation Jacobian (m × N).
      * @param R           Measurement noise covariance (m × m).
      * @return            Normalized Innovation Squared (νᵀ·S⁻¹·ν ~ χ²(m)).
      *                    Expected value = m. Large value → suspect measurement.
      */
    virtual double update(const Eigen::VectorXd& innovation,
                          const Eigen::MatrixXd& H,
                          const Eigen::MatrixXd& R) = 0;

    /** @brief Access the current error covariance matrix.
      * @return The error covariance matrix P (N × N).
      */
    const Eigen::Matrix<double, N, N>& covariance() const { return P_; }

protected:
    Eigen::Matrix<double, N, N> P_ = Eigen::Matrix<double, N, N>::Zero();
};


// ── ES-EKF concrete types ─────────────────────────────────────────────────

/** @brief Nominal state: position, velocity, orientation. */
struct ESEKFState {
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    mr::Quaternion  q;                            // identity by default
};

/** @brief IMU input: specific force and angular rate, both expressed in the body frame. */
struct ImuInput {
    Eigen::Vector3d specific_force;   // [m/s²]
    Eigen::Vector3d angular_rate;     // [rad/s]
};

/** @brief Filter tuning parameters. */
struct ESEKFConfig {
    double var_imu_f = 0.01;                              // IMU force variance   [m²/s⁴]
    double var_imu_w = 0.01;                              // IMU rate variance    [rad²/s²]
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81); // [m/s²]
};


// ── Error-State EKF ────────────────────────────────────────────────────────
//
// Implements the IMU-driven ES-EKF as derived in:
//   Coursera "State Estimation and Localization for Self-Driving Cars",
//   Module 5.  See es_ekf.ipynb for the reference Python implementation.
//
// Error-state dimension: 9  (δp, δv, δθ).
//
class ESEKF : public KalmanFilterBase<9> {
public:
    ESEKF() = default;

    // ── Initialisation ──────────────────────────────────────────────────

    /** @brief Initializes the filter with the given nominal state, error covariance, and config.
     * @param initial_state  Initial nominal state (position, velocity, orientation).
     * @param initial_cov    Initial error covariance (9×9).
     * @param config         Filter tuning parameters (optional).
     */
    void init(const ESEKFState&                    initial_state,
              const Eigen::Matrix<double, 9, 9>&   initial_cov,
              const ESEKFConfig&                   config = ESEKFConfig{})
    {
        state_  = initial_state;
        P_      = initial_cov;
        config_ = config;

        Q_.setZero();
        Q_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * config_.var_imu_f;
        Q_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * config_.var_imu_w;
    }

    // ── IMU prediction step ─────────────────────────────────────────────
    //
    // Propagates the nominal state with dead-reckoning and advances the
    // error-state covariance through the linearised motion model.
    //
    // Process model:
    //   p_k = p_{k-1} + v_{k-1}·dt + ½·(R·f + g)·dt²
    //   v_k = v_{k-1} + (R·f + g)·dt
    //   q_k = q_{k-1} ⊗ q(ω·dt)        (right-composition)
    //
    // Linearised error-state Jacobian F (9×9):
    //   F = [ I,   I·dt,  -½·R·[f]×·dt²  ]
    //       [ 0,   I,     -R·[f]×·dt      ]
    //       [ 0,   0,      I − [ω]×·dt    ]
    //
    // Noise Jacobian L (9×6):
    //   L = [ ½·R·dt²,   0     ]
    //       [ R·dt,      0     ]
    //       [ 0,         I·dt  ]
    //
    /** @brief Propagate the nominal state and error covariance using IMU measurements.
     * The Equations:
     *   p_k = p_{k-1} + v_{k-1}·dt + ½·(R·f + g)·dt²
     *   v_k = v_{k-1} + (R·f + g)·dt
     *   q_k = q_{k-1} ⊗ q(ω·dt)        (right-composition)
     * @param imu  IMU measurements (specific force and angular rate).
     * @param dt   Time step [s].
     */
    void predict(const ImuInput& imu, double dt) {
        Eigen::Matrix3d R_sb = state_.q.toRotationMatrix();
        Eigen::Vector3d a_world = R_sb * imu.specific_force + config_.gravity;

        // Nominal state propagation
        state_.p += state_.v * dt + 0.5 * a_world * (dt * dt);
        state_.v += a_world * dt;
        mr::Quaternion dq = mr::Quaternion::fromAxisAngle(imu.angular_rate * dt);
        state_.q = (state_.q * dq).normalize();   // q_old ⊗ dq

        // Error-state Jacobian F
        Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
        F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
        F.block<3, 3>(0, 6) = -0.5 * R_sb * mr::VecToSo3(imu.specific_force) * (dt * dt);
        F.block<3, 3>(3, 6) = -R_sb * mr::VecToSo3(imu.specific_force) * dt;
        F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() - mr::VecToSo3(imu.angular_rate) * dt;

        // Noise Jacobian L
        Eigen::Matrix<double, 9, 6> L = Eigen::Matrix<double, 9, 6>::Zero();
        L.block<3, 3>(0, 0) = 0.5 * R_sb * (dt * dt);
        L.block<3, 3>(3, 0) = R_sb * dt;
        L.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;

        // Covariance propagation
        P_ = F * P_ * F.transpose() + L * Q_ * L.transpose();
    }

    // ── Measurement updates ─────────────────────────────────────────────

    /** @brief Measurement update for position sensors (GNSS, LiDAR-as-position).
     * H = [I₃ | 0₃ | 0₃] is built internally.
     * @param z_pos      Measured position (3×1).
     * @param R_sensor   Measurement noise covariance (3×3).
     * @return           NIS (νᵀ·S⁻¹·ν ~ χ²(3)). Expected ≈ 3 for a healthy sensor.
     * Note: the caller can use the convenience overload with isotropic noise:
     *       updatePosition(z_pos, sensor_var)  where R_sensor = I₃ * sensor_var.
     */
    double updatePosition(const Eigen::Vector3d& z_pos,
                          const Eigen::Matrix3d& R_sensor)
    {
        Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        Eigen::Vector3d innovation = z_pos - state_.p;
        return applyUpdate(innovation, H, R_sensor);
    }

    /** @brief Convenience overload for position update with isotropic noise.
     * @param z_pos      Measured position (3×1).
     * @param sensor_var Scalar variance for isotropic noise.
     * @return           NIS (νᵀ·S⁻¹·ν ~ χ²(3)). Expected ≈ 3 for a healthy sensor.
     */
    double updatePosition(const Eigen::Vector3d& z_pos, double sensor_var) {
        return updatePosition(z_pos, Eigen::Matrix3d::Identity() * sensor_var);
    }

    /** @brief Generic measurement update for arbitrary sensors.
     * @param innovation  Measurement innovation (z − h(x̂)) pre-computed by the caller.
     * @param H           Observation Jacobian mapping error state to measurements (m×9).
     * @param R           Measurement noise covariance (m×m).
     * @return            NIS (νᵀ·S⁻¹·ν ~ χ²(m)). Expected value = m (measurement dimension).
     */
    double update(const Eigen::VectorXd& innovation,
                  const Eigen::MatrixXd& H,
                  const Eigen::MatrixXd& R) override
    {
        return applyUpdate(innovation, H, R);
    }

    // ── Accessors ───────────────────────────────────────────────────────

    const ESEKFState& state() const { return state_; }

    // Convenience accessors
    const Eigen::Vector3d& position()    const { return state_.p; }
    const Eigen::Vector3d& velocity()    const { return state_.v; }
    const mr::Quaternion&  orientation() const { return state_.q; }

private:
    ESEKFState             state_;
    Eigen::Matrix<double, 6, 6> Q_ = Eigen::Matrix<double, 6, 6>::Zero();
    ESEKFConfig            config_;

    // Compute Kalman gain, inject correction into nominal state, update P. Returns NIS.
    double applyUpdate(const Eigen::VectorXd& innovation,
                       const Eigen::MatrixXd& H,
                       const Eigen::MatrixXd& R)
    {
        auto result = ekfMeasurementUpdate<9>(P_, innovation, H, R);
        injectErrorState(result.dx);
        return result.nis;
    }

    /** @brief Inject the error-state correction dx into the nominal state.
     * The error state is defined as δx = [δp, δv, δθ], where δθ is the
     * orientation error in axis-angle form.  The nominal state is updated as:
     *   p⁺ = p + δp
     *   v⁺ = v + δv
     *   q⁺ = q ⊗ q(δθ)  (right-composition)
     * @param dx  Error-state correction vector (9×1) in the order [δp, δv, δθ].
     */
    void injectErrorState(const Eigen::Matrix<double, 9, 1>& dx) {
        state_.p += dx.segment<3>(0);
        state_.v += dx.segment<3>(3);
        mr::Quaternion dq = mr::Quaternion::fromAxisAngle(dx.segment<3>(6));
        state_.q = (state_.q * dq).normalize();
    }
};

} // namespace kf

#endif // KALMAN_FILTER_H
