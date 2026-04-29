## README.md

# Error-State Extended Kalman Filter (ES-EKF) for IMU-Centric Navigation

A high-performance C++ implementation of an **Error-State Extended Kalman Filter (ES-EKF)** designed for robotics and autonomous systems[cite: 1]. This library provides a robust framework for fusing IMU measurements with external positioning sensors to estimate a robot's 10-DOF state[cite: 1].

The implementation follows the error-state kinematics approach, where the filter estimates the error in the state rather than the full state directly, ensuring better numerical stability and avoiding singularities in orientation[cite: 1].

---

## Key Features

*   **IMU-Centric Prediction**: Propagates state using specific force and angular rate[cite: 1].
*   **10-DOF Nominal State**: Tracks position (3), velocity (3), and orientation (4, unit quaternion)[cite: 1].
*   **9-DOF Error State**: Linearized error tracking for position (3), velocity (3), and orientation error (3, axis-angle)[cite: 1].
*   **Outlier Rejection**: Built-in **Normalized Innovation Squared (NIS)** calculation with Chi-squared ($\chi^2$) thresholding[cite: 1].
*   **Visualization**: Integrated utilities for **CSV** export, **MATLAB/Octave** script generation, and live **Gnuplot** rendering[cite: 2].

---

## Library Components

### 1. `kalman_filter.h`
The core implementation of the filter logic[cite: 1].

*   **`kf::ESEKF`**: The main class for the Error-State EKF[cite: 1].
*   **`kf::ekfMeasurementUpdate<N>`**: A reusable free function for the EKF update step (Gain, Innovation, and Covariance update)[cite: 1].
*   **`kf::chi2Threshold(m, confidence)`**: Provides $\chi^2$ quantiles for outlier detection[cite: 1].

### 2. `kf_graph_utils.h`
Snapshot and visualization utilities[cite: 2].

*   **`kf::FilterRecorder`**: Collects `FilterSnapshot` data during runtime[cite: 2].
*   **`toMatlab(path)`**: Generates a self-contained script for plotting states with $\pm2\sigma$ confidence bands[cite: 2].
*   **`updateLivePlot()`**: Refreshes a live Gnuplot window during execution[cite: 2].

---

## Guidance & Function Reference

### Initialization
Before running the filter, you must initialize it with a starting state and an initial covariance matrix.

```cpp
kf::ESEKF filter;
kf::ESEKFState init_state; // Default: zero pos/vel, identity quaternion
Eigen::Matrix<double, 9, 9> init_P = Eigen::Matrix<double, 9, 9>::Identity() * 0.1;

filter.init(init_state, init_P);
```

### Prediction Step
Call `predict()` every time a new IMU measurement is received.

*   **`predict(ImuInput imu, double dt)`**: Propagates the state[cite: 1].
    *   `imu.specific_force`: Accelerometer reading in body frame[cite: 1].
    *   `imu.angular_rate`: Gyroscope reading in body frame[cite: 1].

### Measurement Update
When an external sensor (like GNSS) provides data, update the filter.

*   **`updatePosition(z_pos, R_sensor)`**: Convenience for 3D position sensors[cite: 1].
*   **`update(innovation, H, R)`**: Generic update for any sensor measurement[cite: 1].
    *   `innovation`: $z - h(x)$ (The difference between measured and predicted state)[cite: 1].
    *   `H`: Observation Jacobian mapping error state to measurement[cite: 1].

### Outlier Detection
To prevent "bad" data from corrupting the filter, check the NIS value returned by update functions.

```cpp
double nis = filter.updatePosition(gnss_pos, 0.05);
if (kf::isOutlier(nis, 3)) { // 3 degrees of freedom for position
    // Reject measurement or log warning
}
```

---

## File Structure

*   **`kalman_filter.h`**: ES-EKF implementation, state structures, and Chi-squared logic[cite: 1].
*   **`kf_graph_utils.h`**: Logging, CSV/Matlab export, and Gnuplot interfacing[cite: 2].</N>