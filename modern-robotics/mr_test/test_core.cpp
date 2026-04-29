// g++ -I /usr/include/eigen3 test/test_core.cpp -o test/test_core

#include <iostream>
#include <iomanip>
#include "../include/mr/core.h"
#include "../include/mr/utils.h"

int main() {
    std::cout << "Starting Modern Robotics Core Library Test..." << std::endl << std::endl;

    // --- בדיקה 1: SO(3) ---
    std::cout << "Test 1: SO(3) Exponential and Logarithm" << std::endl;
    Eigen::Vector3d omega_theta(0, 0, M_PI / 2.0); 
    std::cout << "Input Vector (omega * theta): " << omega_theta.transpose() << std::endl;

    Eigen::Matrix3d so3mat = mr::VecToSo3(omega_theta);
    Eigen::Matrix3d R = mr::MatrixExp3(so3mat);
    mr::printMatrix("Generated Rotation Matrix (R)", R);

    Eigen::Matrix3d so3mat_log = mr::MatrixLog3(R);
    Eigen::Vector3d omg_recovered = mr::So3ToVec(so3mat_log);
    std::cout << "Recovered Vector: " << omg_recovered.transpose() << std::endl;
    std::cout << "Error: " << (omega_theta - omg_recovered).norm() << std::endl << std::endl;


    // --- בדיקה 2: SE(3) Pure Translation ---
    std::cout << "Test 2: SE(3) Pure Translation" << std::endl;
    Eigen::Matrix<double, 6, 1> V_trans;
    V_trans << 0, 0, 0, 1.0, 2.0, 3.0; 
    std::cout << "Input Twist (V): " << V_trans.transpose() << std::endl;

    Eigen::Matrix4d T_trans = mr::MatrixExp6(mr::VecToSe3(V_trans));
    mr::printMatrix("Generated Transformation Matrix (T)", T_trans);

    Eigen::Matrix4d se3mat_log_trans = mr::MatrixLog6(T_trans);
    Eigen::VectorXd V_rec_trans = mr::Se3ToVec(se3mat_log_trans);
    std::cout << "Recovered Twist: " << V_rec_trans.transpose() << std::endl;
    std::cout << "Error: " << (V_trans - V_rec_trans).norm() << std::endl << std::endl;


    // --- בדיקה 3: SE(3) Screw Motion ---
    std::cout << "Test 3: SE(3) Combined Rotation and Translation" << std::endl;
    Eigen::Matrix<double, 6, 1> V_screw;
    V_screw << 0, 0, M_PI, 0, 0, 1.0; 
    std::cout << "Input Screw Twist (V): " << V_screw.transpose() << std::endl;

    Eigen::Matrix4d T_screw = mr::MatrixExp6(mr::VecToSe3(V_screw));
    mr::printMatrix("Generated Screw Transformation (T)", T_screw);

    Eigen::Matrix4d se3mat_log_screw = mr::MatrixLog6(T_screw);
    Eigen::VectorXd V_rec_screw = mr::Se3ToVec(se3mat_log_screw);
    std::cout << "Recovered Screw Twist: " << V_rec_screw.transpose() << std::endl;
    std::cout << "Error: " << (V_screw - V_rec_screw).norm() << std::endl << std::endl;

    // --- בדיקה 5: Screw to Axis ו- AxisAng6 ---
    std::cout << "Test 5: Screw Axis Decomposition" << std::endl;
    Eigen::Vector3d q(1, 0, 0); // נקודה על הציר
    Eigen::Vector3d s(0, 0, 1); // כיוון הציר (Z)
    double h = 2.0;             // Pitch (עלייה ליניארית לסיבוב)

    Eigen::VectorXd S = mr::ScrewToAxis(q, s, h);
    std::cout << "Generated Screw Axis S: " << S.transpose() << std::endl;

    // ננסה להזיז ב-theta = PI
    Eigen::VectorXd expc6 = S * M_PI;
    auto [S_rec, theta_rec] = mr::AxisAng6(expc6);

    std::cout << "Recovered theta: " << theta_rec << " (Expected: " << M_PI << ")" << std::endl;
    std::cout << "Error in S: " << (S - S_rec).norm() << std::endl << std::endl;

    // --- בדיקה 6: Matrix Inversion ---
    std::cout << "Test 6: Transformation Inverse" << std::endl;
    Eigen::Matrix4d T_inv = mr::TransInv(T_screw); // T_screw מהבדיקה הקודמת
    Eigen::Matrix4d Identity_check = T_screw * T_inv;
    mr::printMatrix("T * T_inv (Should be Identity)", Identity_check);

    std::cout << "Test Suite Completed." << std::endl;
    return 0;
}