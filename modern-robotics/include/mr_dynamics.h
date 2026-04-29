#ifndef MR_DYNAMICS_H
#define MR_DYNAMICS_H

#include <Eigen/Dense>
#include <vector>
#include "mr_core.h"

namespace mr {
namespace dynamics {

    // =========================================================================
    // DYNAMICS OF OPEN CHAINS  (Modern Robotics, Chapter 8 & 11)
    //
    // Convention note
    // ---------------
    // mr::adVec(V) implements  -ad(V)^T  (i.e. [[w],[v]; 0,[w]]).
    // The textbook identities therefore become:
    //   ad(V)   = -adVec(V)^T
    //   ad(V)^T = -adVec(V)
    //
    // Data layout
    // -----------
    //   Mlist  : std::vector<Eigen::Matrix4d>          size n+1
    //            Mlist[i] is the home transform of frame {i} w.r.t. frame {i-1}.
    //   Glist  : std::vector<Eigen::Matrix<double,6,6>> size n
    //            Glist[i] is the spatial inertia of link i.
    //   Slist  : Eigen::MatrixXd  (6 x n)
    //            Columns are the joint screw axes in the space frame.
    // =========================================================================


    // -------------------------------------------------------------------------
    // InverseDynamics
    // -------------------------------------------------------------------------

    /**
     * @brief Computes joint forces/torques via Newton-Euler inverse dynamics.
     *
     * Solves: taulist = M(theta)*ddtheta + c(theta,dtheta) + g(theta) + J^T*Ftip
     * using a forward-backward Newton-Euler recursion.
     *
     * Logic & Formula:
     * 1. Forward pass – propagate body velocities and accelerations from the base
     *    outward to the tip.
     *    V_{i+1}  = AdT_i * V_i  + A_i * dtheta_i
     *    Vd_{i+1} = AdT_i * Vd_i + A_i * ddtheta_i - adVec(V_{i+1})^T * A_i * dtheta_i
     *              (the Coriolis term uses adVec(V)^T = -ad(V))
     * 2. Backward pass – propagate spatial forces from the tip back to the base.
     *    F_i   = AdT_{i+1}^T * F_{i+1} + G_i * Vd_{i+1} + adVec(V_{i+1}) * G_i * V_{i+1}
     *              (adVec(V) = -ad(V)^T replaces the textbook  -ad(V)^T  term)
     *    tau_i = F_i^T * A_i
     *
     * Example Input (3-link robot):
     *   thetalist   = {0.1, 0.1, 0.1}
     *   dthetalist  = {0.1, 0.2, 0.3}
     *   ddthetalist = {2,   1.5, 1  }
     *   g           = {0, 0, -9.8}
     *   Ftip        = {1, 1, 1, 1, 1, 1}
     *   (see MATLAB reference for Mlist / Glist / Slist)
     * Expected output:
     *   taulist ≈ {74.6962, -33.0677, -3.2306}
     *
     * @param thetalist   n-vector of joint variables.
     * @param dthetalist  n-vector of joint rates.
     * @param ddthetalist n-vector of joint accelerations.
     * @param g           3-vector for gravitational acceleration.
     * @param Ftip        Spatial force applied by the end-effector in frame {n+1}.
     * @param Mlist       n+1 home-position link frames.
     * @param Glist       n spatial inertia matrices.
     * @param Slist       6×n screw-axis matrix.
     * @return            n-vector of required joint forces/torques.
     */
    inline Eigen::VectorXd InverseDynamics(
        const Eigen::VectorXd& thetalist,
        const Eigen::VectorXd& dthetalist,
        const Eigen::VectorXd& ddthetalist,
        const Eigen::Vector3d& g,
        const Eigen::Matrix<double, 6, 1>& Ftip,
        const std::vector<Eigen::Matrix4d>& Mlist,
        const std::vector<Eigen::Matrix<double, 6, 6>>& Glist,
        const Eigen::MatrixXd& Slist)
    {
        int n = thetalist.size();

        Eigen::Matrix4d Mi = Eigen::Matrix4d::Identity();
        Eigen::MatrixXd Ai(6, n);
        std::vector<Eigen::Matrix<double, 6, 6>> AdTi(n + 1);

        // Vi / Vdi : columns 0..n, column 0 is the base frame.
        Eigen::MatrixXd Vi  = Eigen::MatrixXd::Zero(6, n + 1);
        Eigen::MatrixXd Vdi = Eigen::MatrixXd::Zero(6, n + 1);

        // Gravity appears as a fictitious upward linear acceleration at the base.
        Vdi.col(0).tail(3) = -g;

        // End-effector frame: needed at the start of the backward pass.
        AdTi[n] = Adjoint(TransInv(Mlist[n]));

        Eigen::Matrix<double, 6, 1> Fi = Ftip;
        Eigen::VectorXd taulist(n);

        // Forward pass: propagate velocities and accelerations base → tip.
        for (int i = 0; i < n; ++i) {
            Mi = Mi * Mlist[i];
            Ai.col(i) = Adjoint(TransInv(Mi)) * Slist.col(i);

            // AdT_i = Adj(Exp(-A_i * theta_i) * T_i^{-1})
            Eigen::VectorXd screw_neg = -Ai.col(i) * thetalist(i);
            AdTi[i] = Adjoint(MatrixExp6(VecToSe3(screw_neg)) * TransInv(Mlist[i]));

            Vi.col(i + 1) = AdTi[i] * Vi.col(i)
                            + Ai.col(i) * dthetalist(i);

            // Coriolis term: ad(V)*A*dtheta = -adVec(V)^T * A * dtheta
            Eigen::Matrix<double, 6, 1> Vi_new = Vi.col(i + 1);
            Vdi.col(i + 1) = AdTi[i] * Vdi.col(i)
                             + Ai.col(i) * ddthetalist(i)
                             - adVec(Vi_new).transpose() * Ai.col(i) * dthetalist(i);
        }

        // Backward pass: propagate forces/torques tip → base.
        for (int i = n - 1; i >= 0; --i) {
            // -ad(V)^T*(G*V) = adVec(V)*(G*V)
            Eigen::Matrix<double, 6, 1> Vi_i1 = Vi.col(i + 1);
            Fi = AdTi[i + 1].transpose() * Fi
                 + Glist[i] * Vdi.col(i + 1)
                 + adVec(Vi_i1) * (Glist[i] * Vi_i1);
            taulist(i) = Fi.dot(Ai.col(i));
        }

        return taulist;
    }


    // -------------------------------------------------------------------------
    // MassMatrix
    // -------------------------------------------------------------------------

    /**
     * @brief Computes the numerical mass (inertia) matrix M(thetalist) of an n-joint serial chain.
     *
     * Calls InverseDynamics n times, each time with a unit acceleration on one joint
     * and all other inputs (gravity, tip force, velocities) set to zero.
     * Each call produces one column of M.
     *
     * Example Input (3-link robot):
     *   thetalist = {0.1, 0.1, 0.1}
     * Expected output:
     *   M ≈ [[22.5433, -0.3071, -0.0072],
     *         [-0.3071,  1.9685,  0.4322],
     *         [-0.0072,  0.4322,  0.1916]]
     *
     * @param thetalist n-vector of joint variables.
     * @param Mlist     n+1 home-position link frames.
     * @param Glist     n spatial inertia matrices.
     * @param Slist     6×n screw-axis matrix.
     * @return          n×n symmetric positive-definite inertia matrix.
     */
    inline Eigen::MatrixXd MassMatrix(
        const Eigen::VectorXd& thetalist,
        const std::vector<Eigen::Matrix4d>& Mlist,
        const std::vector<Eigen::Matrix<double, 6, 6>>& Glist,
        const Eigen::MatrixXd& Slist)
    {
        int n = thetalist.size();
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);

        const Eigen::Vector3d             g_zero     = Eigen::Vector3d::Zero();
        const Eigen::Matrix<double, 6, 1> Ftip_zero  = Eigen::Matrix<double, 6, 1>::Zero();
        const Eigen::VectorXd             dtheta_zero = Eigen::VectorXd::Zero(n);

        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd ddthetalist = Eigen::VectorXd::Zero(n);
            ddthetalist(i) = 1.0;
            M.col(i) = InverseDynamics(thetalist, dtheta_zero, ddthetalist,
                                       g_zero, Ftip_zero, Mlist, Glist, Slist);
        }
        return M;
    }


    // -------------------------------------------------------------------------
    // VelQuadraticForces
    // -------------------------------------------------------------------------

    /**
     * @brief Computes the Coriolis and centripetal joint forces/torques c(theta, dtheta).
     *
     * Calls InverseDynamics with zero gravity, zero end-effector force, and zero
     * joint accelerations; the result is the velocity-quadratic term of the
     * equations of motion.
     *
     * Example Input (3-link robot):
     *   thetalist  = {0.1, 0.1, 0.1}
     *   dthetalist = {0.1, 0.2, 0.3}
     * Expected output:
     *   c ≈ {0.2645, -0.0551, -0.0069}
     *
     * @param thetalist  n-vector of joint variables.
     * @param dthetalist n-vector of joint rates.
     * @param Mlist      n+1 home-position link frames.
     * @param Glist      n spatial inertia matrices.
     * @param Slist      6×n screw-axis matrix.
     * @return           n-vector of Coriolis/centripetal joint forces/torques.
     */
    inline Eigen::VectorXd VelQuadraticForces(
        const Eigen::VectorXd& thetalist,
        const Eigen::VectorXd& dthetalist,
        const std::vector<Eigen::Matrix4d>& Mlist,
        const std::vector<Eigen::Matrix<double, 6, 6>>& Glist,
        const Eigen::MatrixXd& Slist)
    {
        int n = thetalist.size();
        return InverseDynamics(thetalist, dthetalist,
                               Eigen::VectorXd::Zero(n),
                               Eigen::Vector3d::Zero(),
                               Eigen::Matrix<double, 6, 1>::Zero(),
                               Mlist, Glist, Slist);
    }


    // -------------------------------------------------------------------------
    // GravityForces
    // -------------------------------------------------------------------------

    /**
     * @brief Computes the joint forces/torques required to overcome gravity at thetalist.
     *
     * Calls InverseDynamics with zero joint velocities, zero accelerations, and
     * zero end-effector force; only gravity acts on the chain.
     *
     * Example Input (3-link robot):
     *   thetalist = {0.1, 0.1, 0.1}
     *   g         = {0, 0, -9.8}
     * Expected output:
     *   grav ≈ {28.4033, -37.6409, -5.4416}
     *
     * @param thetalist n-vector of joint variables.
     * @param g         3-vector for gravitational acceleration.
     * @param Mlist     n+1 home-position link frames.
     * @param Glist     n spatial inertia matrices.
     * @param Slist     6×n screw-axis matrix.
     * @return          n-vector of gravity-compensation joint forces/torques.
     */
    inline Eigen::VectorXd GravityForces(
        const Eigen::VectorXd& thetalist,
        const Eigen::Vector3d& g,
        const std::vector<Eigen::Matrix4d>& Mlist,
        const std::vector<Eigen::Matrix<double, 6, 6>>& Glist,
        const Eigen::MatrixXd& Slist)
    {
        int n = thetalist.size();
        return InverseDynamics(thetalist,
                               Eigen::VectorXd::Zero(n),
                               Eigen::VectorXd::Zero(n),
                               g,
                               Eigen::Matrix<double, 6, 1>::Zero(),
                               Mlist, Glist, Slist);
    }


    // -------------------------------------------------------------------------
    // EndEffectorForces
    // -------------------------------------------------------------------------

    /**
     * @brief Computes the joint forces/torques required solely to produce a given
     *        end-effector spatial force Ftip (J^T * Ftip).
     *
     * Calls InverseDynamics with zero gravity, zero joint velocities, and zero
     * joint accelerations; only the tip wrench acts on the chain.
     *
     * Example Input (3-link robot):
     *   thetalist = {0.1, 0.1, 0.1}
     *   Ftip      = {1, 1, 1, 1, 1, 1}
     * Expected output:
     *   JTFtip ≈ {1.4095, 1.8577, 1.3924}
     *
     * @param thetalist n-vector of joint variables.
     * @param Ftip      Spatial force applied by the end-effector in frame {n+1}.
     * @param Mlist     n+1 home-position link frames.
     * @param Glist     n spatial inertia matrices.
     * @param Slist     6×n screw-axis matrix.
     * @return          n-vector of joint forces/torques due to Ftip.
     */
    inline Eigen::VectorXd EndEffectorForces(
        const Eigen::VectorXd& thetalist,
        const Eigen::Matrix<double, 6, 1>& Ftip,
        const std::vector<Eigen::Matrix4d>& Mlist,
        const std::vector<Eigen::Matrix<double, 6, 6>>& Glist,
        const Eigen::MatrixXd& Slist)
    {
        int n = thetalist.size();
        return InverseDynamics(thetalist,
                               Eigen::VectorXd::Zero(n),
                               Eigen::VectorXd::Zero(n),
                               Eigen::Vector3d::Zero(),
                               Ftip,
                               Mlist, Glist, Slist);
    }


    // -------------------------------------------------------------------------
    // ForwardDynamics
    // -------------------------------------------------------------------------

    /**
     * @brief Computes joint accelerations given applied torques and the current state.
     *
     * Solves:  M(theta) * ddtheta = tau - c(theta,dtheta) - g(theta) - J^T*Ftip
     * using an LDLT (Cholesky) factorisation of the positive-definite mass matrix.
     *
     * Example Input (3-link robot):
     *   thetalist  = {0.1, 0.1, 0.1}
     *   dthetalist = {0.1, 0.2, 0.3}
     *   taulist    = {0.5, 0.6, 0.7}
     *   g          = {0, 0, -9.8}
     *   Ftip       = {1, 1, 1, 1, 1, 1}
     * Expected output:
     *   ddthetalist ≈ {-0.9739, 25.5847, -32.9150}
     *
     * @param thetalist  n-vector of joint variables.
     * @param dthetalist n-vector of joint rates.
     * @param taulist    n-vector of applied joint forces/torques.
     * @param g          3-vector for gravitational acceleration.
     * @param Ftip       Spatial force applied by the end-effector in frame {n+1}.
     * @param Mlist      n+1 home-position link frames.
     * @param Glist      n spatial inertia matrices.
     * @param Slist      6×n screw-axis matrix.
     * @return           n-vector of resulting joint accelerations.
     */
    inline Eigen::VectorXd ForwardDynamics(
        const Eigen::VectorXd& thetalist,
        const Eigen::VectorXd& dthetalist,
        const Eigen::VectorXd& taulist,
        const Eigen::Vector3d& g,
        const Eigen::Matrix<double, 6, 1>& Ftip,
        const std::vector<Eigen::Matrix4d>& Mlist,
        const std::vector<Eigen::Matrix<double, 6, 6>>& Glist,
        const Eigen::MatrixXd& Slist)
    {
        Eigen::VectorXd rhs = taulist
                              - VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
                              - GravityForces(thetalist, g, Mlist, Glist, Slist)
                              - EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist);

        return MassMatrix(thetalist, Mlist, Glist, Slist).ldlt().solve(rhs);
    }


    // -------------------------------------------------------------------------
    // InverseDynamicsTrajectory
    // -------------------------------------------------------------------------

    /**
     * @brief Computes joint forces/torques for an entire trajectory via inverse dynamics.
     *
     * Iterates over each time step and calls InverseDynamics to compute the
     * required torques at each configuration along the trajectory.
     *
     * @param thetamat   N×n matrix of joint angles (one row per time step).
     * @param dthetamat  N×n matrix of joint velocities.
     * @param ddthetamat N×n matrix of joint accelerations.
     * @param g          3-vector for gravitational acceleration.
     * @param Ftipmat    N×6 matrix of end-effector spatial forces.
     * @param Mlist      n+1 home-position link frames.
     * @param Glist      n spatial inertia matrices.
     * @param Slist      6×n screw-axis matrix.
     * @return           N×n matrix of joint forces/torques (one row per time step).
     */
    inline Eigen::MatrixXd InverseDynamicsTrajectory(
        const Eigen::MatrixXd& thetamat,
        const Eigen::MatrixXd& dthetamat,
        const Eigen::MatrixXd& ddthetamat,
        const Eigen::Vector3d& g,
        const Eigen::MatrixXd& Ftipmat,
        const std::vector<Eigen::Matrix4d>& Mlist,
        const std::vector<Eigen::Matrix<double, 6, 6>>& Glist,
        const Eigen::MatrixXd& Slist)
    {
        int N = thetamat.rows();
        int n = thetamat.cols();
        Eigen::MatrixXd taumat(N, n);

        for (int i = 0; i < N; ++i) {
            Eigen::Matrix<double, 6, 1> Ftip = Ftipmat.row(i).transpose();
            taumat.row(i) = InverseDynamics(
                thetamat.row(i).transpose(),
                dthetamat.row(i).transpose(),
                ddthetamat.row(i).transpose(),
                g, Ftip, Mlist, Glist, Slist).transpose();
        }
        return taumat;
    }


    // -------------------------------------------------------------------------
    // ForwardDynamicsTrajectory
    // -------------------------------------------------------------------------

    /**
     * @brief Simulates a serial chain's motion under an open-loop torque history.
     *
     * Integrates the equations of motion forward in time using a simple Euler scheme.
     * For each time step, ForwardDynamics is called intRes times within the interval
     * dt, each sub-step advancing the state by dt/intRes.
     *
     * @param thetalist  n-vector of initial joint variables.
     * @param dthetalist n-vector of initial joint rates.
     * @param taumat     N×n matrix of applied joint forces/torques (one row per step).
     * @param g          3-vector for gravitational acceleration.
     * @param Ftipmat    N×6 matrix of end-effector spatial forces.
     * @param Mlist      n+1 home-position link frames.
     * @param Glist      n spatial inertia matrices.
     * @param Slist      6×n screw-axis matrix.
     * @param dt         Time step between consecutive torque rows.
     * @param intRes     Number of Euler sub-steps per time step (≥ 1).
     * @return           A pair {thetamat, dthetamat}, each N×n (one row per step).
     */
    inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ForwardDynamicsTrajectory(
        const Eigen::VectorXd& thetalist,
        const Eigen::VectorXd& dthetalist,
        const Eigen::MatrixXd& taumat,
        const Eigen::Vector3d& g,
        const Eigen::MatrixXd& Ftipmat,
        const std::vector<Eigen::Matrix4d>& Mlist,
        const std::vector<Eigen::Matrix<double, 6, 6>>& Glist,
        const Eigen::MatrixXd& Slist,
        double dt,
        int intRes)
    {
        int N = taumat.rows();
        int n = taumat.cols();
        Eigen::MatrixXd thetamat_out(N, n);
        Eigen::MatrixXd dthetamat_out(N, n);

        Eigen::VectorXd theta  = thetalist;
        Eigen::VectorXd dtheta = dthetalist;
        thetamat_out.row(0)  = theta.transpose();
        dthetamat_out.row(0) = dtheta.transpose();

        double sub_dt = dt / intRes;

        for (int i = 0; i < N - 1; ++i) {
            Eigen::Matrix<double, 6, 1> Ftip = Ftipmat.row(i).transpose();
            Eigen::VectorXd tau = taumat.row(i).transpose();

            for (int j = 0; j < intRes; ++j) {
                Eigen::VectorXd ddtheta = ForwardDynamics(
                    theta, dtheta, tau, g, Ftip, Mlist, Glist, Slist);
                // Euler step: theta and dtheta use eulerIntegration from mr_core
                theta  = eulerIntegration(theta,  dtheta,  sub_dt);
                dtheta = eulerIntegration(dtheta, ddtheta, sub_dt);
            }
            thetamat_out.row(i + 1)  = theta.transpose();
            dthetamat_out.row(i + 1) = dtheta.transpose();
        }
        return {thetamat_out, dthetamat_out};
    }


    // -------------------------------------------------------------------------
    // ComputedTorque
    // -------------------------------------------------------------------------

    /**
     * @brief Computes joint torques using a feedback-linearising (computed-torque) controller.
     *
     * Combines PID error feedback with an inverse-dynamics feedforward term:
     *   tau = M(theta) * (Kp*e + Ki*(eint + e) + Kd*(dthetad - dtheta))
     *         + InverseDynamics(theta, dtheta, ddthetad, g, 0)
     *
     * where e = thetad - theta is the joint position error.
     * Kp, Ki, Kd are scalar gains applied identically to all joints.
     *
     * Example Input (3-link robot):
     *   thetalist   = {0.1, 0.1, 0.1}
     *   dthetalist  = {0.1, 0.2, 0.3}
     *   eint        = {0.2, 0.2, 0.2}
     *   g           = {0, 0, -9.8}
     *   thetalistd  = {1, 1, 1}
     *   dthetalistd = {2, 1.2, 2}
     *   ddthetalistd= {0.1, 0.1, 0.1}
     *   Kp=1.3, Ki=1.2, Kd=1.1
     * Expected output:
     *   taulist ≈ {133.0053, -29.9422, -3.0328}
     *
     * @param thetalist    Current joint angles.
     * @param dthetalist   Current joint velocities.
     * @param eint         Time-integral of joint position errors.
     * @param g            3-vector for gravitational acceleration.
     * @param Mlist        n+1 home-position link frames.
     * @param Glist        n spatial inertia matrices.
     * @param Slist        6×n screw-axis matrix.
     * @param thetalistd   Reference joint angles.
     * @param dthetalistd  Reference joint velocities.
     * @param ddthetalistd Reference joint accelerations (feedforward).
     * @param Kp           Proportional gain (scalar, same for all joints).
     * @param Ki           Integral gain (scalar, same for all joints).
     * @param Kd           Derivative gain (scalar, same for all joints).
     * @return             n-vector of commanded joint torques.
     */
    inline Eigen::VectorXd ComputedTorque(
        const Eigen::VectorXd& thetalist,
        const Eigen::VectorXd& dthetalist,
        const Eigen::VectorXd& eint,
        const Eigen::Vector3d& g,
        const std::vector<Eigen::Matrix4d>& Mlist,
        const std::vector<Eigen::Matrix<double, 6, 6>>& Glist,
        const Eigen::MatrixXd& Slist,
        const Eigen::VectorXd& thetalistd,
        const Eigen::VectorXd& dthetalistd,
        const Eigen::VectorXd& ddthetalistd,
        double Kp, double Ki, double Kd)
    {
        Eigen::VectorXd e = thetalistd - thetalist;

        Eigen::VectorXd pid_accel = Kp * e
                                  + Ki * (eint + e)
                                  + Kd * (dthetalistd - dthetalist);

        return MassMatrix(thetalist, Mlist, Glist, Slist) * pid_accel
               + InverseDynamics(thetalist, dthetalist, ddthetalistd,
                                 g, Eigen::Matrix<double, 6, 1>::Zero(),
                                 Mlist, Glist, Slist);
    }


} // namespace dynamics
} // namespace mr


#endif
