#ifndef MR_UTILS_H
#define MR_UTILS_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace mr {

    /* --- IGNORE ---
     * @brief Utility function to print matrices in a formatted way for debugging.
     * This function is not part of the core library but is useful for testing and visualization.
     * It prints the name of the matrix followed by its contents with fixed precision.
     * @param name A string label for the matrix being printed.
     * @param mat The Eigen matrix to be printed.
     */
    void printMatrix(const std::string& name, const Eigen::MatrixXd& mat) {
    std::cout << "--- " << name << " ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4) << mat << std::endl << std::endl;
    }

    /** --- IGNORE ---
     * @brief Utility function to print a header for test cases.
     * @param msg The message to be printed as a header.
     */
    void printHeader(const std::string& msg) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << msg << "\n";
    std::cout << std::string(60, '=') << "\n";
    }



}


#endif