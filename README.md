# robotics-utils

A collection of utility libraries for robotics applications.

## Libraries

- [kalman-filter](#kalman-filter)
- [modern-robotics](#modern-robotics)

---

## kalman-filter

A library for state estimation using the Kalman Filter algorithm, useful for filtering noisy sensor data in robotics systems.

### Installation

```bash
npm install @robotics-utils/kalman-filter
```

### Usage

```typescript
import { KalmanFilter } from '@robotics-utils/kalman-filter';

const kf = new KalmanFilter({
  // configuration options
});

const estimate = kf.update(measurement);
```

### Features

- Linear Kalman Filter
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Supports multi-dimensional state estimation

---

## modern-robotics

A library based on the **Modern Robotics** textbook (Lynch & Park), providing tools for robot kinematics, dynamics, and motion planning.

### Installation

```bash
npm install @robotics-utils/modern-robotics
```

### Usage

```typescript
import { } from '@robotics-utils/modern-robotics';
```

### Features

- Forward & Inverse Kinematics
- Screw theory and Lie group operations
- Transformation matrices (SO3, SE3)
- Jacobian computation
- Trajectory generation

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## License

MIT
