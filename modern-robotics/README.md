# ModernRobotics C++ Library

A high-performance C++ library for robot kinematics, dynamics, and motion planning based on the "Modern Robotics" framework by Lynch and Park. This library leverages **Eigen3** for linear algebra and provides robust implementations for Lie group theory ($SE(3)$, $SO(3)$), serial chain kinematics, and advanced path planning solvers[cite: 8, 10, 11].

## 🚀 Core Modules

### 1. Mathematical Foundation (`mr_core.h`)
The heart of the library, providing Lie Algebra operations and rigid-body transformation utilities[cite: 8]:
*   **Vector/Matrix Conversions**: Functions like `VecToSo3` and `VecToSe3` to map vectors to skew-symmetric and $se(3)$ matrices[cite: 8].
*   **Matrix Exponential/Logarithm**: Core algorithms for $SO(3)$ and $SE(3)$ to convert between velocities (twists) and configurations (poses)[cite: 8].
*   **Adjoint Transformations**: 6x6 Adjoint matrices for transforming twists and wrenches between frames[cite: 8].
*   **Numerical Stability**: Includes thresholds for near-zero values and robust pseudo-inverse calculations using SVD[cite: 8].

### 2. Kinematics (`mr_kinematics.h`)
Tools for analyzing serial chain mechanisms[cite: 10]:
*   **Forward Kinematics**: PoE (Product of Exponentials) formula implementations in both space (`FKinSpace`) and body (`FKinBody`) frames[cite: 10].
*   **Jacobian Calculations**: Computes Space and Body Jacobians that map joint velocities to end-effector twists[cite: 10].
*   **Inverse Kinematics**: Robust solvers using Newton-Raphson or Damped Least Squares (DLS) for stability near singularities[cite: 10].

### 3. Motion Planning (`mr_motion_planning.h`)
A decoupled planning architecture that separates map representation from solver logic[cite: 11]:
*   **Solvers**: Includes **RRT*** (asymptotically optimal), **PRM** (Probabilistic Roadmap), and Grid-based **A***[cite: 11].
*   **Map Representations**: Support for 2D Occupancy Grids with distance transforms for obstacle inflation and analytic circular obstacle maps[cite: 11].
*   **Collision Checking**: Abstract `IWorldMap` interface allowing users to swap sensor modalities (LiDAR, Camera) without changing solvers[cite: 11].

### 4. Trajectory Generation (`mr_trajectory.h`)
Generates smooth motion profiles for joint and task space[cite: 13]:
*   **Time Scaling**: Supports Cubic and Quintic time scaling for smooth acceleration and deceleration[cite: 13].
*   **Interpolation**: Generates **Screw Trajectories** (straight lines in $SE(3)$) and **Cartesian Trajectories** (decoupled rotation and translation)[cite: 13].

### 5. Odometry (`mr_odometry.h`)
Implements configuration updates for mobile bases[cite: 12]:
*   **Omnidirectional Update**: Computes the next robot state ($x, y, \phi$) using exact integration of body twists[cite: 12].
*   **Transformation Updates**: Utilities to update 4x4 $T_{sb}$ matrices based on robot displacement[cite: 12].

---

## 🛠 File Overview

| File | Primary Responsibility |
| :--- | :--- |
| **`mr_core.h`** | Lie Algebra ($so3, se3$), Adjoints, Matrix Exp/Log, Pseudo-inverse[cite: 8]. |
| **`mr_kinematics.h`** | Forward/Inverse kinematics, Body/Space Jacobians, Singularity detection[cite: 10]. |
| **`mr_motion_planning.h`**| RRT*, PRM, A* algorithms, and Grid/Circular map interfaces[cite: 11]. |
| **`mr_trajectory.h`** | Smooth path interpolation and time scaling (Cubic/Quintic)[cite: 13]. |
| **`mr_odometry.h`** | Mobile base state updates and $SE(2)$ integration[cite: 12]. |
| **`mr_utils.h`** | Formatted matrix printing and test debugging utilities[cite: 14]. |
| **`mr_dynamics.h`** | Placeholder for future mass matrix and Coriolis force calculations[cite: 9]. |

---

## 💻 Quick Start

### Example: Forward Kinematics
```cpp
#include <mr_kinematics.h>

// Define home configuration and screw axes
Eigen::Matrix4d M = ...; 
Eigen::MatrixXd Blist = ...;
Eigen::VectorXd thetalist = ...;

// Compute pose
Eigen::Matrix4d T = mr::kinematics::FKinBody(M, Blist, thetalist);
```

### Example: Path Planning
```cpp
#include "mr_motion_planning.h"

// 1. Setup Map and Params
mr::planning::map::OccupancyGrid grid(100, 100, 0.05);
mr::planning::OccupancyGridMap world(grid, 0.1); // 10cm robot radius

// 2. Plan Path
mr::planning::Path path = mr::planning::PlanPath(world, start, goal);
```

## 📋 Dependencies
*   **Eigen 3.3+**: Required for all matrix operations[cite: 8, 10].
*   **C++11/14/17**: Compatible with modern C++ standards.