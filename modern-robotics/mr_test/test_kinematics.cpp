#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include "../include/mr/kinematics.h"
#include "../include/mr/utils.h"


int main() {
    int n = 2; // רובוט RR
    
    // 1. הגדרת מערכת המרחב {s} והבית M
    Eigen::MatrixXd Slist(6, n);
    Slist << 0, 0,
             0, 0,
             1, 1,
             0, 0,
             0, -1, 
             0, 0;

    Eigen::Matrix4d M;
    M << 1, 0, 0, 2,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;

    // 2. חישוב אוטומטי של Blist כדי להבטיח עקביות מתמטית
    // Bi = Ad(M^-1) * Si
    Eigen::MatrixXd Blist(6, n);
    for (int i = 0; i < n; ++i) {
        Blist.col(i) = mr::Adjoint(mr::TransInv(M)) * Slist.col(i);
    }

    mr::printMatrix("Slist (Space Frame)", Slist);
    mr::printMatrix("Blist (Computed for consistency)", Blist);

    // --- טסט 1: בדיקת Adjoint ו-Jacobian (הטסט שנכשל) ---
    mr::printHeader("Test 1: Adjoint Consistency (Js == Ad(T) * Jb)");
    Eigen::VectorXd test_theta(n);
    test_theta << M_PI / 4.0, M_PI / 6.0;

    Eigen::Matrix4d T_sb = mr::FKinSpace(M, Slist, test_theta);
    Eigen::MatrixXd Js = mr::JacobianSpace(Slist, test_theta);
    Eigen::MatrixXd Jb = mr::JacobianBody(Blist, test_theta);

    Eigen::MatrixXd Js_from_Ad = mr::Adjoint(T_sb) * Jb;

    mr::printMatrix("Js (Direct)", Js);
    mr::printMatrix("Js (Calculated via Adjoint)", Js_from_Ad);

    double jac_err = (Js - Js_from_Ad).norm();
    std::cout << "Jacobian Error: " << jac_err << std::endl;
    if (jac_err < 1e-8) std::cout << "Result: PASSED ✅ (Adjoint is fixed!)\n";
    else std::cout << "Result: FAILED ❌ (Check Adjoint in core.h)\n";


    // --- טסט 2: IK Loop - יעד בר השגה ---
    mr::printHeader("Test 2: IK Reachable Target Loop");
    Eigen::Matrix4d T_target = T_sb; // היעד שהגענו אליו ב-FK
    Eigen::VectorXd guess(n);
    guess << 0.7, 0.5; // ניחוש קרוב

    auto ik_res = mr::IKinBody(T_target, M, Blist, guess, 1e-5, 1e-5);
    
    if (ik_res.second) {
        std::cout << "IK Success! Angles found: " << ik_res.first.transpose() << std::endl;
        double angle_err = (ik_res.first - test_theta).norm();
        std::cout << "Angle Difference from original: " << angle_err << std::endl;
        std::cout << "Result: PASSED ✅\n";
    } else {
        std::cout << "Result: FAILED ❌ (IK should have converged)\n";
    }


    // --- טסט 3: IK Reachability - יעד רחוק מדי ---
    mr::printHeader("Test 3: Unreachable Target (Outside Workspace)");
    Eigen::Matrix4d T_impossible = M;
    T_impossible(0, 3) = 10.0; // ננסה להגיע ל-10 מטר כשהזרוע באורך 2 מטר

    auto ik_fail_res = mr::IKinBody(T_impossible, M, Blist, guess, 1e-3, 1e-3);
    
    if (!ik_fail_res.second) {
        std::cout << "IK correctly identified target as unreachable. ✅\n";
        std::cout << "Best angles found: " << ik_fail_res.first.transpose() << std::endl;
    } else {
        std::cout << "Something is wrong: IK claimed to reach an impossible point! ❌\n";
    }

    return 0;
}