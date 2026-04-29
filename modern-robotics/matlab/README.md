# Modern Robotics Matlab Scripts

This repository contains a comprehensive collection of MATLAB functions developed for robotic analysis, kinematics, dynamics, and control. These scripts are based on the algorithms and mathematical foundations presented in **"Modern Robotics: Mechanics, Planning, and Control"** by Kevin M. Lynch and Frank C. Park.

## 📁 Repository Structure

The scripts cover several core areas of robotics:

### 1. Rigidbody Motion & Transformations
Functions for handling rotations ($SO(3)$) and rigid-body motions ($SE(3)$).
* `AxisAng3.m` / `AxisAng6.m`: Convert between exponential coordinates and axis-angle representations.
* `VecToso3.m` / `so3ToVec.m`: Matrix-vector conversions for rotations.
* `VecTose3.m` / `se3ToVec.m`: Matrix-vector conversions for rigid-body transformations.
* `Adjoint.m` / `ad.m`: Adjoint maps for twists and wrenches.
* `TransInv.m` / `TransToRp.m`: Transformation matrix utilities.

### 2. Forward & Inverse Kinematics
Tools for calculating end-effector positions and joint configurations.
* `FKinSpace.m` / `FKinBody.m`: Forward kinematics in space and body frames.
* `IKinBody.m` / `IKinBodyIterates.m`: Numerical inverse kinematics using Newton-Raphson.
* `JacobianSpace.m` / `JacobianBody.m`: Analytic Jacobian calculations.

### 3. Dynamics & Control
Functions for modeling robot physics and implementing control loops.
* `ForwardDynamics.m` / `InverseDynamics.m`: Newton-Euler and Lagrangian dynamics.
* `GravityForces.m` / `VelQuadraticForces.m`: Component-wise dynamic forces.
* `ComputedTorque.m` / `SimulateControl.m`: Feedback control laws and simulation routines.
* `EulerStep.m`: Numerical integration for dynamic simulation.

### 4. Trajectory Generation
* `CubicTimeScaling.m` / `QuinticTimeScaling.m`: Smooth motion profiles.
* `CartesianTrajectory.m` / `ScrewTrajectory.m`: Path planning in task space.
* `JointTrajectory.m`: Path planning in configuration space.

### 5. Specialized Analysis
* `FormClosure.m` / `AssemblyForceClosure.m`: Grasping and contact mechanics analysis.
* `FindMaximuLinearSpeedForWheelRobot.m`: Mobile robot kinematics.
* `RunFullAssemblyAnalysis.m`: High-level script for evaluating mechanical assemblies.

## 🚀 Getting Started

### Prerequisites
* MATLAB (R2021a or later recommended).
* Basic understanding of Lie Algebra and Screw Theory (optional but helpful).

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/ShaiPollak/Modern-Robotics-Matlab-Scripts.git](https://github.com/ShaiPollak/Modern-Robotics-Matlab-Scripts.git)
