#ifndef MR_MOTION_PLANNING_H
#define MR_MOTION_PLANNING_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/* Motion planning namespace for the ModernRobotics library.
 *
 * Architecture overview:
 *
 *   mr::planning::map         — Map representations (OccupancyGrid)
 *   IWorldMap                 — Abstract collision-checking interface
 *   mr::planning::solvers     — Planning algorithms (RRT*, PRM, A*)
 *   PlanPath()                — Top-level entry point
 *
 * The planning algorithms are completely decoupled from sensor types.
 * Swap camera → LiDAR → depth sensor by providing a different IWorldMap
 * implementation; the solvers remain untouched.
 *
 * State space:
 *   State = Eigen::VectorXd  (joint angles for arms, [x,y] for mobile bases)
 *   Path  = std::vector<State>
 *
 * Typical workflow:
 * @code
 *   // 1. Build a map
 *   mr::planning::map::OccupancyGrid grid(200, 200, 0.01);
 *   grid.setCellState(gx, gy, mr::planning::map::CellState::Occupied);
 *
 *   // 2. Wrap it in a world-map
 *   mr::planning::OccupancyGridMap world(grid, 0.05);   // 5 cm robot radius
 * 
 *   // 3. (Optional) Tune planning parameters
 *   mr::planning::PlanningParams params = 
 *          mr::planning::configuratePathPlanningParams(0.05, 0.10, 5000);
 *   
 *   // 4. Plan
 *   mr::planning::State start(2); 
 *   start << -0.4, -0.4;                               //(X, Y) start position in world coordinates
 *   mr::planning::Goal  goal;  
 *   goal.state.resize(2); goal.state << 0.4, 0.4;      //(X, Y) goal position in world coordinates
 *   mr::planning::Path  path = mr::planning::PlanPath(params, world, start, goal);
 * @endcode
 */

namespace mr {
namespace planning {

// ============================================================
// FUNDAMENTAL TYPES
// ============================================================

/** @brief Configuration-space state vector (joint angles, 2D/3D pose, …). */
using State = Eigen::VectorXd;

/** @brief An ordered sequence of collision-free states. */
using Path  = std::vector<State>;

// ============================================================
// ROBOT MODEL
// ============================================================

/** @brief A collision sphere attached to a robot link, expressed in the link's local frame. */
struct CollisionSphere {
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    double          radius{0.0};
};

/** @brief Per-DOF joint limits and optional FK model for 3-D arm collision checking. */
struct RobotModel {
    int             dof{0};
    Eigen::VectorXd qMin, qMax;

    /** @brief Per-link collision spheres (index i → link i). */
    std::vector<std::vector<CollisionSphere>> linkSpheres;

    /** @brief Maps a joint config to per-link SE(3) transforms.
     *  Leave unset for 2-D point-robot planning. */
    std::function<std::vector<Eigen::Matrix4d>(const State&)> forwardKinematics;

    bool hasFKModel() const noexcept {
        return static_cast<bool>(forwardKinematics);
    }

    bool withinLimits(const State& q) const noexcept {
        return (q.array() >= qMin.array()).all() &&
               (q.array() <= qMax.array()).all();
    }

    State clamp(const State& q) const noexcept {
        return q.cwiseMax(qMin).cwiseMin(qMax);
    }
};

// ============================================================
// PLANNING DATA STRUCTURES
// ============================================================

/** @brief Target configuration + acceptance tolerance. */
struct Goal {
    State  state;
    double tolerance{1e-3};

    bool reached(const State& q) const noexcept {
        return (q - state).norm() < tolerance;
    }
};

/** @brief Tuning parameters shared across all solvers. */
struct PlanningParams {
    int    maxIterations{5000};

    /** @brief Maximum extension step per iteration (RRT*, PRM edge length). */
    double stepSize{0.05};

    /** @brief Probability of sampling the goal directly (RRT*). */
    double goalBias{0.10};

    /** @brief Number of nearest neighbours to connect in PRM. */
    int    kNeighbors{10};

    /** @brief RRT* rewire radius override.  0 = use the asymptotically optimal formula. */
    double rewireRadius{0.0};

    /** @brief Interpolation step for continuous collision checking. */
    double collisionCheckStep{0.01};
};

// ============================================================
// MAP NAMESPACE
// ============================================================

namespace map {

/** @brief Cell classification for a 2-D occupancy grid. 
 * Occupied means the cell contains an obstacle.  
 * Free means the cell is known to be empty.  
 * Unknown means the cell's state is unknown (e.g., outside sensor range) and should be treated as occupied for safety.
*/
enum class CellState : uint8_t { Free = 0, Occupied = 1, Unknown = 2 };

/** @brief 2-D occupancy grid with optional Euclidean distance transform.
 * State = [x, y] in world coordinates.  Cell (gx, gy) is occupied if its state is Occupied or Unknown.
 * The grid is axis-aligned and supports arbitrary resolution and origin offset.
 * @param width, height  Number of cells in x and y directions.
 * @param resolution     Cell size in metres.
 * @param origin         World coordinates of the grid's (0, 0) cell corner. Defaults to (0, 0).
 *
 * Coordinates:
 *   World (wx, wy) ↔ Grid cell (gx, gy) via  gx = (wx - origin.x) / resolution.
 *
 * Distance transform:
 *   Call computeDistanceTransform() once after the grid is populated. Which computes the Euclidean distance (in metres) 
 *   from each cell to the nearest occupied cell.
 *   distanceToNearestObstacle() then returns the Euclidean distance (in metres)
 *   to the nearest occupied cell — useful for safety-margin costs. */
class OccupancyGrid {
public:
    OccupancyGrid(
        int width, 
        int height, 
        double resolution,
        Eigen::Vector2d origin = Eigen::Vector2d::Zero()): 
        width_(width), 
        height_(height), 
        resolution_(resolution), 
        origin_(origin),
        cells_(width * height, CellState::Free),
        distTransform_(width * height, std::numeric_limits<double>::infinity()
    ) {}

    // ------------------------------------------------------------------
    // Cell access
    // ------------------------------------------------------------------

    /** @brief Sets the state (Occupied, Free, Unknown) of a grid cell.  Out-of-bounds cells are ignored. 
     *  @param gx, gy  Grid cell coordinates.  Cell (0, 0) is at the origin; cell (1, 0) is one cell to the right, etc.
     *  @param s        Cell state to set.  Occupied and Unknown cells are treated as obstacles by the solvers.
    */
    void setCellState(int gx, int gy, CellState s) {
        if (!inBounds(gx, gy)) return;
        cells_[idx(gx, gy)] = s;
        distTransformValid_ = false;
    }

    /** @brief Gets the state (Occupied, Free, Unknown) of a grid cell.  Out-of-bounds cells return Unknown.
     *  @param gx, gy  Grid cell coordinates.  Cell (0, 0) is at the origin; cell (1, 0) is one cell to the right, etc.
     *  @return Cell state. */
    CellState getCellState(int gx, int gy) const noexcept {
        if (!inBounds(gx, gy)) return CellState::Unknown;
        return cells_[idx(gx, gy)];
    }

    /** @brief Returns true if (gx, gy) is a valid cell coordinate.
     *  @param gx, gy  Grid cell coordinates.  Cell (0, 0) is at the origin; cell (1, 0) is one cell to the right, etc.
     *  @return True if (gx, gy) is within the grid bounds. */
    bool inBounds(int gx, int gy) const noexcept {
        return gx >= 0 && gx < width_ && gy >= 0 && gy < height_;
    }

    // ------------------------------------------------------------------
    // Coordinate conversions
    // ------------------------------------------------------------------

    /** @brief World → grid cell index (truncates; may return out-of-bounds cell). 
     * @param p  World coordinates (in metres).
     * @return Grid cell coordinates index. */
    Eigen::Vector2i worldToGrid(const Eigen::Vector2d& p) const noexcept {
        return {static_cast<int>((p.x() - origin_.x()) / resolution_),
                static_cast<int>((p.y() - origin_.y()) / resolution_)};
    }

    /** @brief Grid cell centre → world coordinates. 
     * @param gx, gy  Grid cell coordinates index.  Cell (0, 0) is at the origin; cell (1, 0) is one cell to the right, etc.
     * @return World coordinates (in metres) of the cell centre. */
    Eigen::Vector2d gridToWorld(int gx, int gy) const noexcept {
        return {origin_.x() + (gx + 0.5) * resolution_,
                origin_.y() + (gy + 0.5) * resolution_};
    }

    // ------------------------------------------------------------------
    // Occupancy query
    // ------------------------------------------------------------------

    /** @brief Returns true if the cell containing world point p is occupied or unknown.
     * @param p  World coordinates (in metres).
     * @return True if the cell containing p is occupied or unknown. */
    bool isOccupied(const Eigen::Vector2d& p) const noexcept {
        auto g = worldToGrid(p);
        return !inBounds(g.x(), g.y()) ||
               cells_[idx(g.x(), g.y())] == CellState::Occupied;
    }

    /** @brief Returns true if the cell containing world point p is free.
     * @param p  World coordinates (in metres).
     * @return True if the cell containing p is free. */
    bool isFree(const Eigen::Vector2d& p) const noexcept {
        auto g = worldToGrid(p);
        return inBounds(g.x(), g.y()) &&
               cells_[idx(g.x(), g.y())] == CellState::Free;
    }

    // ------------------------------------------------------------------
    // Distance transform  (lazy, computed on first query after modification)
    // ------------------------------------------------------------------

    /** @brief Multi-source BFS distance transform.  O(W×H) time.
     *  After calling this, distanceToNearestObstacle() is available.
     *  This methods calculates the Euclidean distance (in metres) from each cell to the nearest occupied cell using a 
     *  multi-source breadth-first search (BFS) algorithm. This works like a wavefront expansion from all occupied cells simultaneously, 
     *  efficiently computing the distance to the nearest obstacle for every cell in the grid.
     *  The steps are as follows:
     *  1. Initialize a queue with all occupied cells (distance = 0) and all other cells (distance = ∞).
     *  2. Pop the front cell, and for each of its 8-connected neighbours, if the new distance through the popped cell 
     *  is smaller, update the neighbour's distance and push it to the back of the queue.  Repeat until the queue is empty.
     *  3. Convert cell-unit distances to metres by multiplying by the resolution.
     * */
    void computeDistanceTransform() const {
        std::fill(distTransform_.begin(), distTransform_.end(),
                  std::numeric_limits<double>::infinity()); // reset distances
        std::queue<int> q;                                  
        for (int i = 0; i < width_ * height_; ++i) {
            if (cells_[i] == CellState::Occupied) {
                distTransform_[i] = 0.0; // occupied cells have zero distance to the nearest obstacle (themselves)
                q.push(i);
            }
        }
        // 8-connected BFS in cell units, converted to metres at the end
        const int  dx[] = { 1,-1, 0, 0, 1,-1, 1,-1};
        const int  dy[] = { 0, 0, 1,-1, 1,-1,-1, 1};                
        const double dc[] = {1,1,1,1,1.4142,1.4142,1.4142,1.4142};  // Diagonal moves cost sqrt(2), side and up/down moves cost 1
        while (!q.empty()) {
            int cur = q.front(); q.pop();                           // Get the next cell index from the queue
            int gx = cur % width_, gy = cur / width_;               // Convert the cell index to grid coordinates, gx = column index, gy = row index
            for (int d = 0; d < 8; ++d) {                           // Explore all 8-connected neighbours
                int nx = gx + dx[d], ny = gy + dy[d];               // Compute the neighbour's grid coordinates
                if (!inBounds(nx, ny)) continue;
                int  ni      = idx(nx, ny);                         // Convert the neighbour's grid coordinates back to a cell index
                double newDist = distTransform_[cur] + dc[d];       
                
                // If the new distance through the current cell is smaller, update the neighbour's distance and push it to the queue
                // This makes the algorithm work in O(W×H) time, as each cell is updated at most once with its final distance value.
                if (newDist < distTransform_[ni]) { 
                    distTransform_[ni] = newDist;
                    q.push(ni);
                }
            }
        }
        for (auto& v : distTransform_) v *= resolution_; // convert to metres
        distTransformValid_ = true;
    }

    /** @brief Euclidean distance (metres) from grid cell to nearest obstacle.
     *  Auto-computes the distance transform on first call. 
     * @param gx, gy  Grid cell coordinates.  Cell (0, 0) is at the origin; cell (1, 0) is one cell to the right, etc.
     * @return Distance (in metres) from the cell to the nearest occupied cell.
     * */
    double distanceToNearestObstacle(int gx, int gy) const {
        if (!distTransformValid_) computeDistanceTransform();
        if (!inBounds(gx, gy)) return 0.0;
        return distTransform_[idx(gx, gy)];
    }

    /** @brief Euclidean distance (metres) from world point to nearest obstacle.
     *  Auto-computes the distance transform on first call.
     * @param p  World coordinates (in metres).
     * @return Distance (in metres) from the cell to the nearest occupied cell.
     */
    double distanceToNearestObstacle(const Eigen::Vector2d& p) const {
        auto g = worldToGrid(p);
        return distanceToNearestObstacle(g.x(), g.y());
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /** @brief Grid width (number of cells in x direction). */
    int width()       const noexcept { return width_; }
    /** @brief Grid height (number of cells in y direction). */
    int height()      const noexcept { return height_; }
    /** @brief Grid resolution (metres per cell). */
    double resolution()const noexcept { return resolution_; }
    /** @brief Grid origin (world coordinates of cell (0, 0)). */
    const Eigen::Vector2d& origin() const noexcept { return origin_; }

private:
    int width_, height_;
    double resolution_;
    Eigen::Vector2d origin_;
    std::vector<CellState> cells_;
    mutable std::vector<double> distTransform_;
    mutable bool distTransformValid_{false};

    // Helper: converts grid coordinates to cell index in the 1-D cells_ vector.
    int idx(int gx, int gy) const noexcept { return gy * width_ + gx; }
};

} // namespace map

// ============================================================
// WORLD MAP INTERFACE
// ============================================================

/** @brief Abstract interface that decouples map representation from planning algorithm.
 *
 * Implement this interface to connect any sensor modality (camera, LiDAR, depth
 * sensor) to the planning solvers without modifying the solvers themselves.
 * A concrete implementation receives the map data at construction time and
 * exposes only the two collision-query functions the solvers need. */
class IWorldMap {
public:
    // Virtual is used here to allow for polymorphic deletion via base pointers, even though we don't expect any data members in the base class.
    virtual ~IWorldMap() = default;

    /** @brief Returns true if state q is collision-free. */
    virtual bool isCollisionFree(const State& q) const = 0;

    /** @brief Returns true if the linearly interpolated path from → to is collision-free.
     *  @param step  Interpolation resolution (metres or radians). */
    virtual bool isPathCollisionFree(const State& from, const State& to,
                                     double step = 0.01) const = 0;

    /** @brief Axis-aligned bounding box of the valid state space {lower, upper}. */
    virtual std::pair<State, State> getBounds() const = 0;

    /** @brief Dimensionality of the state space. */
    virtual int getDimension() const = 0;

protected:
    /** @brief Helper: walks the segment from → to and returns false on first collision. */
    bool checkSegment(const State& from, const State& to, double step) const {
        double len = (to - from).norm();
        if (len < 1e-9) return isCollisionFree(from);
        int steps = std::max(1, static_cast<int>(std::ceil(len / step)));
        for (int i = 0; i <= steps; ++i) {
            double t = static_cast<double>(i) / steps;
            if (!isCollisionFree(from + t * (to - from))) return false;
        }
        return true;
    }
};

// ------------------------------------------------------------------
// Concrete: OccupancyGrid-backed 2-D world map
// ------------------------------------------------------------------

/** @brief IWorldMap implementation backed by a 2-D OccupancyGrid.
 *
 * State = [x, y] in world coordinates.
 * robotRadius inflates the obstacles: a cell is treated as occupied if the
 * distance transform reports a clearance smaller than robotRadius. */
class OccupancyGridMap : public IWorldMap {
public:
    /** @param grid        Reference to an OccupancyGrid (must outlive this object).
     *  @param robotRadius Robot footprint radius for obstacle inflation. */
    OccupancyGridMap(
        const map::OccupancyGrid& grid, 
        double robotRadius = 0.0): 
        grid_(grid), 
        robotRadius_(robotRadius) {}

    /** @brief Returns true if state q is collision-free, meaning it is not occupied and respects the robot's radius. */
    bool isCollisionFree(const State& q) const override {
        if (q.size() < 2) return false;
        Eigen::Vector2d p(q[0], q[1]);
        if (grid_.isOccupied(p)) return false;
        if (robotRadius_ > 0.0 &&
            grid_.distanceToNearestObstacle(p) < robotRadius_) return false;
        return true;
    }

    /** @brief Returns true if the linearly interpolated path from → to is collision-free, meaning all interpolated points are collision-free.
     *  Checks the path at intervals of step (metres). */
    bool isPathCollisionFree(const State& from, const State& to,
                             double step = 0.01) const override {
        return checkSegment(from, to, step);
    }

    std::pair<State, State> getBounds() const override {
        const auto& orig = grid_.origin();
        double res = grid_.resolution();
        State lo(2), hi(2);
        lo << orig.x(), orig.y();
        hi << orig.x() + grid_.width()  * res,
              orig.y() + grid_.height() * res;
        return {lo, hi};
    }

    int getDimension() const override { return 2; }

private:
    const map::OccupancyGrid& grid_;
    double robotRadius_;
};

// ------------------------------------------------------------------
// Concrete: analytic 2-D circular obstacles (no rasterisation needed)
// ------------------------------------------------------------------

/** @brief Lightweight IWorldMap for a set of 2-D circular obstacles.
 *
 * Useful for rapid prototyping and unit tests without building a full grid.
 * State = [x, y].  Robot is modelled as a point inflated by robotRadius. */
class CircularObstacleMap : public IWorldMap {
public:
    struct Circle { double x, y, radius; };

    /** @param obstacles  List of circular obstacles, defined by their centre (x, y) and radius.  
     *                    The robot's radius is added to each obstacle's radius for collision checking.
     *  @param xMin, xMax Bounds of the valid state space in x.
     *  @param yMin, yMax Bounds of the valid state space in y.
     *  @param robotRadius Robot footprint radius for obstacle inflation. */
    CircularObstacleMap(
        std::vector<Circle> obstacles,
        double xMin, double xMax,
        double yMin, double yMax,
        double robotRadius = 0.0)
        : obstacles_(std::move(obstacles)),
          xMin_(xMin), 
          xMax_(xMax), 
          yMin_(yMin), 
          yMax_(yMax),
          robotRadius_(robotRadius) 
    {
        for (auto& o : obstacles_) o.radius += robotRadius_;
    }

    bool isCollisionFree(const State& q) const override {
        if (q.size() < 2) return false;
        for (const auto& o : obstacles_) {
            double dx = q[0] - o.x, dy = q[1] - o.y;
            if (std::hypot(dx, dy) < o.radius) return false;
        }
        return true;
    }

    bool isPathCollisionFree(
        const State& from, 
        const State& to,
        double step = 0.01) const override 
    {
        // Exact segment-vs-circle test (no rasterisation)
        Eigen::Vector2d ab(to[0] - from[0], to[1] - from[1]);
        double abLen2 = ab.squaredNorm();
        if (abLen2 < 1e-12) return isCollisionFree(from);
        for (const auto& o : obstacles_) {
            Eigen::Vector2d ao(o.x - from[0], o.y - from[1]);
            double t = std::max(0.0, std::min(1.0, ao.dot(ab) / abLen2));
            if ((ao - t * ab).norm() < o.radius) return false;
        }
        return true;
    }

    std::pair<State, State> getBounds() const override {
        State lo(2), hi(2);
        lo << xMin_, yMin_;
        hi << xMax_, yMax_;
        return {lo, hi};
    }

    int getDimension() const override { return 2; }

private:
    std::vector<Circle> obstacles_;
    double xMin_, xMax_, yMin_, yMax_, robotRadius_;

};

// ============================================================
// SOLVERS NAMESPACE
// ============================================================

namespace solvers {

// ------------------------------------------------------------------
// RRT* — Rapidly-exploring Random Tree (optimal variant)
// ------------------------------------------------------------------

/** @brief RRT* solver for any IWorldMap.
 *
 * Runs for params.maxIterations iterations, continuously improving the path
 * cost via the rewiring step.  Returns the best path to the goal found
 * within the iteration budget, or an empty Path if unreachable.
 *
 * Cost propagation to descendants after rewiring is omitted for efficiency;
 * this does not affect asymptotic optimality but slows convergence slightly. */
class RRTStar {
public:
    /** @brief Constructor.
     * @param params  Tuning parameters (see PlanningParams). */
    explicit RRTStar(PlanningParams params = {})
        : params_(std::move(params)) {}

    /** @brief Sets the random seed for reproducibility 
     *  @param seed  Random seed for the internal RNG.  
     *  Call this before solve() to get deterministic behaviour. */
    void setSeed(unsigned seed) noexcept { rng_.seed(seed); }

    /** @brief Plans a path using RRT*.
     *  The Algorithm proceeds as follows:
     * 1. Sample a random state qRand from the state space with goal bias (i.e., with probability params_.goalBias, qRand = goal.state).
     * 2. Find the nearest node qNear in the tree to qRand.
     * 3. Steer from qNear towards qRand to get a new state qNew, at most params_.stepSize away from qNear.
     * 4. If the path from qNear to qNew is collision-free, proceed; otherwise, discard qNew and go back to step 1.
     * 5. Find all existing nodes within a radius rr of qNew (where rr is the rewiring radius, which can be set via params_.rewireRadius or 
     *    computed using the asymptotically optimal formula).
     * 6. Choose the best parent (minimum cost) for qNew from the near nodes.
     * 7. Rewire the near nodes to qNew if it improves their cost.
     * 8. If qNew is within the goal tolerance and has a lower cost than the best goal found so far, update the best goal.
     * 9. Repeat for params_.maxIterations iterations.
     *  @param map   World map for collision queries.
     *  @param start Start state.
     *  @param goal  Goal configuration + tolerance.
     *  @return Collision-free path from start to goal, or empty if unreachable within the iteration budget.
     *  @throws std::runtime_error if start is in collision. */
    Path solve(
        const IWorldMap& map, 
        const State& start, 
        const Goal& goal) 
    {
        if (!map.isCollisionFree(start))
            throw std::runtime_error("RRTStar: start state is in collision.");

        auto [lo, hi] = map.getBounds();
        int d = map.getDimension();

        // Flat node storage — avoids heap allocations per node
        struct Node { State state; int parent{-1}; double cost{0.0}; };
        std::vector<Node> nodes;
        nodes.reserve(params_.maxIterations + 1);
        nodes.push_back({start, -1, 0.0});

        // Per-dimension uniform samplers
        std::vector<std::uniform_real_distribution<double>> samplers;
        samplers.reserve(d); // reserve space for d samplers (each dimension has its own range)
        for (int i = 0; i < d; ++i) samplers.emplace_back(lo[i], hi[i]);
        std::uniform_real_distribution<double> bias(0.0, 1.0);

        // Track the best goal node found during the iterations
        int    bestGoalIdx  = -1;
        double bestGoalCost = std::numeric_limits<double>::infinity();

        // Main RRT* loop:
        for (int iter = 0; iter < params_.maxIterations; ++iter) {
            // 1. Sample (with goal bias)
            State qRand(d);
            if (bias(rng_) < params_.goalBias) {
                qRand = goal.state;
            } else {
                for (int i = 0; i < d; ++i) qRand[i] = samplers[i](rng_);
            }

            // 2. Find nearest node
            int nearIdx = nearestIdx(nodes, qRand);

            // 3. Steer towards qRand from qNear, respecting step size
            State qNew = steer(nodes[nearIdx].state, qRand);

            // 4. Collision check
            if (!map.isCollisionFree(qNew)) continue;
            if (!map.isPathCollisionFree(nodes[nearIdx].state, qNew,
                                          params_.collisionCheckStep)) continue;

            // 5. Find near nodes for rewiring. Rewire radius can be set via params_.rewireRadius or 
            // computed using the asymptotically optimal formula: γ * (log(n)/n)^(1/d), 
            // where n is the current number of nodes and d is the state space dimension.
            double rr = rewireRadius(static_cast<int>(nodes.size()), d);
            auto   nearList = nearIndices(nodes, qNew, rr); // indices of nodes within radius rr of qNew

            // 6. Choose best parent (minimum cost)
            int    bestParent = nearIdx;
            double bestCost   = nodes[nearIdx].cost +
                                (qNew - nodes[nearIdx].state).norm();

            // Check if any of the near nodes offer a cheaper path to qNew, and if the path is collision-free
            for (int ni : nearList) {
                double c = nodes[ni].cost + (qNew - nodes[ni].state).norm();
                if (c < bestCost &&
                    map.isPathCollisionFree(nodes[ni].state, qNew,
                                            params_.collisionCheckStep)) {
                    bestCost   = c;
                    bestParent = ni;
                }
            }

            // 7. Add new node
            int newIdx = static_cast<int>(nodes.size());
            nodes.push_back({qNew, bestParent, bestCost});

            // 8. Rewire: check if going through newNode lowers cost for nearby nodes
            for (int ni : nearList) {
                double newCost = bestCost + (nodes[ni].state - qNew).norm();
                if (newCost < nodes[ni].cost &&
                    map.isPathCollisionFree(qNew, nodes[ni].state,
                                            params_.collisionCheckStep)) {
                    nodes[ni].parent = newIdx;
                    nodes[ni].cost   = newCost;
                }
            }

            // 9. Track best goal
            if (goal.reached(qNew) && bestCost < bestGoalCost) {
                bestGoalCost = bestCost;
                bestGoalIdx  = newIdx;
            }
        }

        if (bestGoalIdx < 0) return {};

        // Reconstruct path by walking parent chain
        Path path;
        for (int i = bestGoalIdx; i != -1; i = nodes[i].parent)
            path.push_back(nodes[i].state);
        std::reverse(path.begin(), path.end());
        return path;
    }

private:
    PlanningParams params_;
    std::mt19937   rng_{std::random_device{}()};

    /** @brief Finds the index of the node in nodes that is nearest to state q. 
     * @param nodes  List of nodes to search through. Each node must have a .state member of type State.
     * @param q      Query state.
     * @return Index of the node in nodes that is nearest to q.  Uses squared Euclidean distance for efficiency.
    */
    template<typename NodeT>
    int nearestIdx(
        const std::vector<NodeT>& nodes, 
        const State& q) const noexcept 
    {
        int    best = 0;
        double bestD = std::numeric_limits<double>::infinity();
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            double d = (q - nodes[i].state).squaredNorm();
            if (d < bestD) { bestD = d; best = i; }
        }
        return best;
    }

    /** @brief Finds the indices of all nodes in nodes that are within radius of state q. 
     * @param nodes  List of nodes to search through. Each node must have a .state member of type State.
     * @param q      Query state.
     * @param radius  Radius within which to search for near nodes.
     * @return Vector of indices of nodes in nodes that are within radius of q.  Uses squared Euclidean distance for efficiency.
    */
    template<typename NodeT>
    std::vector<int> nearIndices(
        const std::vector<NodeT>& nodes,
        const State& q, double radius) const 
    {
        std::vector<int> result;
        double r2 = radius * radius;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
            if ((q - nodes[i].state).squaredNorm() <= r2) result.push_back(i);
        return result;
    }

    /** @brief Steers from → to, returning a new state at most params_.stepSize away from from in the direction of to. 
     * If to is within stepSize of from, returns to.  Otherwise, returns from + stepSize * (to - from) / ||to - from||.
     * This is used to incrementally grow the RRT* tree towards random samples while respecting the maximum step size constraint.
     * @param from  Starting state.
     * @param to    Target state.
     * @return New state that is at most stepSize away from from in the direction of to.  
     * If to is within stepSize of from, returns to.  Otherwise, returns from + stepSize * (to - from) / ||to - from||.
    */
    State steer(const State& from, const State& to) const noexcept {
        State diff = to - from;
        double dist = diff.norm();
        if (dist <= params_.stepSize) return to;
        return from + (params_.stepSize / dist) * diff;
    }
    
    /** @brief Computes the rewiring radius for RRT*.
     *  @param n  Number of nodes in the tree.
     *  @param d  Dimension of the state space.
     *  @return Rewiring radius. */
    double rewireRadius(int n, int d) const noexcept {
        if (params_.rewireRadius > 0.0) return params_.rewireRadius;
        if (n <= 1) return params_.stepSize;
        // Asymptotically optimal radius: γ * (log(n)/n)^(1/d)
        const double gamma = 2.0 * params_.stepSize;
        return std::min(gamma * std::pow(std::log(static_cast<double>(n)) / n,
                                         1.0 / d),
                        params_.stepSize * 3.0);
    }
};

// ------------------------------------------------------------------
// Probabilistic Roadmap (PRM)
// ------------------------------------------------------------------

/** @brief Two-phase PRM: build a reusable roadmap, then query it with A*.
 *
 * Phase 1 — buildRoadmap():
 *   Samples collision-free states uniformly and connects each to its
 *   k-nearest collision-free neighbours.  The roadmap persists across queries.
 *
 * Phase 2 — query():
 *   Temporarily attaches start and goal to the roadmap, runs A* on the
 *   resulting graph, and returns the path without permanently modifying
 *   the stored roadmap. */
class ProbabilisticRoadmap {
public:
    explicit ProbabilisticRoadmap(PlanningParams params = {})
        : params_(std::move(params)) {}

    void setSeed(unsigned seed) noexcept { rng_.seed(seed); }
    void clearRoadmap() noexcept { states_.clear(); adj_.clear(); }
    int  roadmapSize() const noexcept { return static_cast<int>(states_.size()); }

    /** @brief Populates the roadmap with numSamples collision-free nodes. 
     * Each new node is connected to its k nearest neighbours if the path is collision-free.
     *  @param map        World map for collision queries.
     *  @param numSamples Number of collision-free samples to add to the roadmap.
    */
    void buildRoadmap(
        const IWorldMap& map, 
        int numSamples = 1000) 
    {
        states_.clear();
        adj_.clear();

        auto [lo, hi] = map.getBounds();
        int d = map.getDimension();

        std::vector<std::uniform_real_distribution<double>> samplers;
        for (int i = 0; i < d; ++i) samplers.emplace_back(lo[i], hi[i]);

        const int maxAttempts = numSamples * 20;
        for (int attempts = 0;
             static_cast<int>(states_.size()) < numSamples && attempts < maxAttempts;
             ++attempts) {
            State q(d);
            for (int i = 0; i < d; ++i) q[i] = samplers[i](rng_);
            if (!map.isCollisionFree(q)) continue;

            int newIdx = static_cast<int>(states_.size());
            states_.push_back(q);
            adj_.push_back({});
            connectNode(newIdx, map);
        }
    }

    /** @brief Returns a collision-free path from start to goal through the roadmap.
     *  Returns an empty Path if no path exists. 
     *  @param map   World map for collision queries.
     *  @param start Start state.
     *  @param goal  Goal state.
     *  @return Collision-free path from start to goal, or empty Path if no path exists.
     */
    Path query(
        const IWorldMap& map, 
        const State& start, 
        const Goal& goal) 
    {
        int n = static_cast<int>(states_.size());

        // Build augmented graph locally (start=n, goal=n+1)
        std::vector<State>                                localStates = states_;
        std::vector<std::vector<std::pair<int,double>>>   localAdj    = adj_;
        localStates.push_back(start);      localAdj.push_back({});
        localStates.push_back(goal.state); localAdj.push_back({});
        int startIdx = n, goalIdx = n + 1;

        auto attachNode = [&](int tempIdx) {
            auto nearest = kNearest(localStates[tempIdx], params_.kNeighbors,
                                    localStates, tempIdx);
            for (int ni : nearest) {
                if (!map.isPathCollisionFree(localStates[tempIdx], localStates[ni],
                                              params_.collisionCheckStep)) continue;
                double c = (localStates[tempIdx] - localStates[ni]).norm();
                localAdj[tempIdx].emplace_back(ni, c);
                localAdj[ni].emplace_back(tempIdx, c);
            }
        };
        attachNode(startIdx);
        attachNode(goalIdx);

        auto idxPath = aStarGraph(startIdx, goalIdx, localStates, localAdj, goal.state);

        Path path;
        path.reserve(idxPath.size());
        for (int i : idxPath) path.push_back(localStates[i]);
        return path;
    }

private:
    PlanningParams params_;
    std::vector<State>                              states_;
    std::vector<std::vector<std::pair<int,double>>> adj_;
    std::mt19937 rng_{std::random_device{}()};

    /** @brief Connects a new node to its k nearest neighbours in the roadmap if the path is collision-free. 
     *  This is called when a new node is added to the roadmap during buildRoadmap(), and also for the temporary start and goal nodes during query().
     *  @param newIdx Index of the new node in states_.
     *  @param map    World map for collision queries.
    */
    void connectNode(
        int newIdx, 
        const IWorldMap& map) 
    {
        auto nearest = kNearest(states_[newIdx], params_.kNeighbors, states_, newIdx);
        for (int ni : nearest) {
            if (!map.isPathCollisionFree(states_[newIdx], states_[ni],
                                          params_.collisionCheckStep)) continue;
            double c = (states_[newIdx] - states_[ni]).norm();
            adj_[newIdx].emplace_back(ni, c);
            adj_[ni].emplace_back(newIdx, c);
        }
    }

    /** @brief Finds the indices of the k nearest nodes in pool to state q, excluding excludeIdx. 
     *  Uses squared Euclidean distance for efficiency.  Returns at most k indices, fewer if pool is smaller. 
     *  @param q          Query state.
     *  @param k          Number of nearest neighbours to return.
     *  @param pool       List of states to search through.
     *  @param excludeIdx Optional index to exclude from consideration (e.g., the new node itself).
     *  @return Vector of indices of the k nearest states in pool to q, excluding excludeIdx. */
    static std::vector<int> kNearest(
        const State& q, int k,
        const std::vector<State>& pool,
        int excludeIdx = -1) 
    {
        std::vector<std::pair<double,int>> ranked;
        ranked.reserve(pool.size());
        for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
            if (i == excludeIdx) continue;
            ranked.emplace_back((q - pool[i]).squaredNorm(), i);
        }
        int take = std::min(k, static_cast<int>(ranked.size()));
        std::partial_sort(ranked.begin(), ranked.begin() + take, ranked.end());
        std::vector<int> result;
        result.reserve(take);
        for (int i = 0; i < take; ++i) result.push_back(ranked[i].second);
        return result;
    }

    /** @brief Runs A* on the given graph to find a path from startIdx to goalIdx. 
     *  The graph is defined by localStates (node states) and localAdj (adjacency list with edge costs). 
     *  Uses the Euclidean distance to the goal as the heuristic. 
     *  Returns a vector of node indices representing the path from startIdx to goalIdx, or an empty vector if no path exists.
     *  @param startIdx   Index of the start node in localStates.
     *  @param goalIdx    Index of the goal node in localStates.
     *  @param localStates List of node states in the graph.
     *  @param localAdj    Adjacency list of the graph, where localAdj[i] is a vector of pairs (neighbor index, edge cost) for node i.
     *  @param goalState   State of the goal node, used for heuristic calculation. */
    static std::vector<int> aStarGraph(
            int startIdx, 
            int goalIdx,
            const std::vector<State>& nodes,
            const std::vector<std::vector<std::pair<int,double>>>& adj,
            const State& goalState) 
    {
        struct NodeData {
            double g{std::numeric_limits<double>::infinity()};
            double f{std::numeric_limits<double>::infinity()};
            int parent{-1};
            bool closed{false};
        };
        std::vector<NodeData> data(nodes.size());
        data[startIdx].g = 0.0;
        data[startIdx].f = (goalState - nodes[startIdx]).norm();

        using PQ = std::pair<double,int>;
        std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> pq;
        pq.push({data[startIdx].f, startIdx});

        while (!pq.empty()) {
            auto [f, cur] = pq.top(); pq.pop();
            if (data[cur].closed) continue;
            data[cur].closed = true;

            if (cur == goalIdx) {
                std::vector<int> path;
                for (int i = cur; i != -1; i = data[i].parent)
                    path.push_back(i);
                std::reverse(path.begin(), path.end());
                return path;
            }

            for (const auto& [nb, cost] : adj[cur]) {
                if (data[nb].closed) continue;
                double newG = data[cur].g + cost;
                if (newG < data[nb].g) {
                    data[nb].g      = newG;
                    data[nb].f      = newG + (goalState - nodes[nb]).norm();
                    data[nb].parent = cur;
                    pq.push({data[nb].f, nb});
                }
            }
        }
        return {};
    }
};

// ------------------------------------------------------------------
// A* on OccupancyGrid
// ------------------------------------------------------------------

/** @brief Grid-based A* on a 2-D OccupancyGrid.
 *
 * State = [x, y] in world coordinates.  Uses 8-connected grid with
 * Euclidean distance heuristic.  Optionally inflates obstacles via
 * the grid's distance transform when robotRadius > 0. */
class AStar {
public:
    /** @brief Plans a path using A* on the given OccupancyGrid.
     *  @returns Returns a path from start to goal, or an empty path if unreachable.
     *  @param grid        OccupancyGrid to plan on.
     *  @param start       Start state [x, y] in world coordinates.
     *  @param goal        Goal configuration + tolerance.
     *  @param robotRadius  Minimum clearance from obstacles (uses distance transform).
     *  @throws std::invalid_argument if start or goal states are not 2-D.
     *  @throws std::runtime_error if start or goal are outside grid bounds. */
    Path solve(
        const map::OccupancyGrid& grid,
        const State& start,
        const Goal&  goal,
        double robotRadius = 0.0) 
    {
        if (start.size() < 2 || goal.state.size() < 2)
            throw std::invalid_argument("AStar: state must be 2-D [x, y].");

        auto sg = grid.worldToGrid(Eigen::Vector2d(start[0],      start[1]));
        auto gg = grid.worldToGrid(Eigen::Vector2d(goal.state[0], goal.state[1]));

        if (!grid.inBounds(sg.x(), sg.y()))
            throw std::runtime_error("AStar: start is outside grid bounds.");
        if (!grid.inBounds(gg.x(), gg.y()))
            throw std::runtime_error("AStar: goal is outside grid bounds.");

        int W = grid.width(), H = grid.height();
        auto cellIdx = [W](int gx, int gy) { return gy * W + gx; };

        struct GridNode {
            double g{std::numeric_limits<double>::infinity()};
            double f{std::numeric_limits<double>::infinity()};
            int    parent{-1};
            bool   closed{false};
        };
        std::vector<GridNode> nodes(W * H);

        double res = grid.resolution();
        auto heuristic = [&](int gx, int gy) {
            return res * std::hypot(gg.x() - gx, gg.y() - gy);
        };

        int sIdx = cellIdx(sg.x(), sg.y());
        int gIdx = cellIdx(gg.x(), gg.y());
        nodes[sIdx].g = 0.0;
        nodes[sIdx].f = heuristic(sg.x(), sg.y());

        using PQ = std::pair<double,int>;
        std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> pq;
        pq.push({nodes[sIdx].f, sIdx});

        const int    dx[] = { 1,-1, 0, 0, 1,-1, 1,-1};
        const int    dy[] = { 0, 0, 1,-1, 1,-1,-1, 1};
        const double dc[] = { 1, 1, 1, 1, 1.4142, 1.4142, 1.4142, 1.4142};

        while (!pq.empty()) {
            auto [f, cur] = pq.top(); pq.pop();
            if (nodes[cur].closed) continue;
            nodes[cur].closed = true;

            if (cur == gIdx) {
                Path path;
                for (int i = cur; i != -1; i = nodes[i].parent) {
                    int gx = i % W, gy = i / W;
                    auto wp = grid.gridToWorld(gx, gy);
                    State s(2); s << wp.x(), wp.y();
                    path.push_back(s);
                }
                std::reverse(path.begin(), path.end());
                return path;
            }

            int curGx = cur % W, curGy = cur / W;
            for (int d = 0; d < 8; ++d) {
                int nx = curGx + dx[d], ny = curGy + dy[d];
                if (!grid.inBounds(nx, ny)) continue;
                if (grid.getCellState(nx, ny) == map::CellState::Occupied) continue;
                if (robotRadius > 0.0 &&
                    grid.distanceToNearestObstacle(nx, ny) < robotRadius) continue;

                int ni = cellIdx(nx, ny);
                if (nodes[ni].closed) continue;

                double newG = nodes[cur].g + dc[d] * res;
                if (newG < nodes[ni].g) {
                    nodes[ni].g      = newG;
                    nodes[ni].f      = newG + heuristic(nx, ny);
                    nodes[ni].parent = cur;
                    pq.push({nodes[ni].f, ni});
                }
            }
        }
        return {};
    }
};

} // namespace solvers

// ============================================================
// TOP-LEVEL ENTRY POINT
// ============================================================

/** @brief Plans a collision-free path using RRT* (default).
 *
 * For grid-based planning prefer solvers::AStar::solve() directly.
 * For multi-query scenarios build a solvers::ProbabilisticRoadmap once and
 * call query() repeatedly.
 *
 * @param map    World model that answers collision queries.
 * @param start  Start configuration.
 * @param goal   Goal configuration + tolerance.
 * @param params Algorithm tuning parameters.
 * @return       Collision-free path start → goal, or empty if unreachable. */
inline Path PlanPath(
    const IWorldMap&       map,
    const State&           start,
    const Goal&            goal,
    const PlanningParams&  params = {}) 
{
    solvers::RRTStar rrt(params);
    return rrt.solve(map, start, goal);
}

/** @brief Helper to set common RRT* parameters in one call. 
 *  @param stepSize      Maximum distance for each tree extension step (metres).
 *  @param goalBias      Probability of sampling the goal state instead of a random state (0.0 to 1.0).
 *  @param maxIterations Maximum number of iterations to run the RRT* algorithm. */
inline PlanningParams configuratePathPlanningParams(
    double stepSize, 
    double goalBias, 
    int maxIterations) 
{
    PlanningParams params;
    params.stepSize = stepSize;
    params.goalBias = goalBias;
    params.maxIterations = maxIterations;
    return params;
}
} // namespace planning
} // namespace mr

#endif // MR_MOTION_PLANNING_H

