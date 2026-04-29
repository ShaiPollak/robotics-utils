from tracemalloc import start

import numpy as np


class C_SPACE:
    """
    C_SPACE class defines the configuration space for the path planning problem. 
    It takes a list of obstacles, the radius of the object, and the range of x and y coordinates. 
    The constructor inflates the obstacles by the radius of the object to ensure that the path 
    planning accounts for the size of the object.
    """

    def __init__(self, obstacles, object_radius, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5)):
        self.dimension = 2
        self.x_range_min = x_range[0]
        self.x_range_max = x_range[1]
        self.y_range_min = y_range[0]
        self.y_range_max = y_range[1]
        self.obstacles = obstacles
        self.object_radius = object_radius

        # Inflate the obstacles by the radius of the object to ensure that the path planning accounts 
        # for the size of the object
        for obstacle in self.obstacles:
            obstacle.radius += self.object_radius

    def get_range(self):
        return (self.x_range_min, self.x_range_max), (self.y_range_min, self.y_range_max)



class Tree(C_SPACE):
    """
    The Tree class implements algorithms for path planning in a 2D C-space with circular obstacles. 
    It inherits from the C_SPACE class, which defines the configuration space and handles obstacle inflation.
    """
    def __init__(self, obstacles, object_radius=0.01, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5)):
        super().__init__(obstacles, object_radius, x_range, y_range)
        self.samples = []

    def get_samples(self):
        return self.samples
    
    # Sample random points in the C-space and build the RRT tree
    def generate_tree(self, start_point, goal_point, num_samples, method, min_distance_to_goal=0.05):
        #Reset the sample count before generating a new tree
        Sample.num_samples = 0 
        if method == 'RRT':
            self.rrt_algorithm(start_point, goal_point, num_samples, min_distance_to_goal=min_distance_to_goal, max_steering_dist=0.05)

        elif method == 'RPM':
            self.rpm_algorithm(start_point, goal_point, num_samples)

        else:
            raise ValueError("Unsupported method. Please choose 'RRT' or 'RPM'.")

    # Check if a sample is inside any of the obstacles
    def sample_inside_obstacles(self, sample):
        for obstacle in self.obstacles:
            distance = sample.get_distance(obstacle)
            if distance < obstacle.radius:
                #print(f"Sample at ({sample.x}, {sample.y}) is inside Obstacle {obstacle.obstacle_id} with center at ({obstacle.x}, {obstacle.y}) and radius {obstacle.radius}.")
                return True
        return False
    
    # Check if the path between two samples collides with any obstacle
    def path_collision_between(self, sample1, sample2):
        # Path vector from sample1 to sample2
        vec_sample1_to_sample2 = np.array([sample2.x - sample1.x, sample2.y - sample1.y])
        
        for obstacle in self.obstacles:
            
            # Vector from sample1 to the obstacle center
            vec_sample1_to_obstacle = np.array([obstacle.x - sample1.x, obstacle.y - sample1.y])
            
            # Project the vector from sample1 to the obstacle onto the path vector
            t = np.dot(vec_sample1_to_obstacle, vec_sample1_to_sample2) / np.dot(vec_sample1_to_sample2, vec_sample1_to_sample2)
            
            # If the projection falls outside the segment, we can ignore this obstacle for collision checking
            if t < 0 or t > 1:
                continue

            # Calculate the distance from the obstacle to the path
            distance_to_path = np.linalg.norm(vec_sample1_to_obstacle - t * vec_sample1_to_sample2)
            
            if distance_to_path < obstacle.radius:
                return True
            
            return False

    # Find the nearest sample in the tree to a given random sample      
    def get_nearest_sample(self, random_sample):
        best_distance = float('inf')
        for sample in self.samples:
            distance = random_sample.get_distance(sample)
            if distance < best_distance:
                best_distance = distance
                closest_sample = sample
        
        return closest_sample
    
    def get_kth_nearest_sample(self, random_sample, k=8):
        close_samples_distances = []
        for sample in self.samples:
            distance = random_sample.get_distance(sample)
            close_samples_distances.append((distance, sample))
        
        close_samples_distances.sort(key=lambda x: x[0])  # Sort by distance
        kth_nearest_samples = [close_samples_distances[i][1] for i in range(min(k, len(close_samples_distances)))]  # Get the k-th nearest samples (1-based index)
        
        return kth_nearest_samples
    
    def steer(self, nearest_sample, random_coordinate, max_distance):
        # Calculate the vector from the nearest sample to the random sample
        vec_nearest_to_random = np.array([random_coordinate.x - nearest_sample.x, random_coordinate.y - nearest_sample.y])
        distance = np.linalg.norm(vec_nearest_to_random)

        if  distance < max_distance:
            return random_coordinate
        else:
            # Scale the vector to have a length of max_distance
            vec_scaled = (vec_nearest_to_random / distance) * max_distance
            random_coordinate.x = nearest_sample.x + vec_scaled[0]
            random_coordinate.y = nearest_sample.y + vec_scaled[1]
            return random_coordinate
    
    # RRT algorithm implementation
    def rrt_algorithm(self, start_point, goal_point, max_samples=1000, min_distance_to_goal=0.05, max_steering_dist=0.05):
        self.samples = []
        Sample.num_samples = 0 # Reset counter

        # 1. Initialize Start and Goal as Coordinates first to avoid wasting IDs
        # Goal is just a reference point, it doesn't need to be in the tree yet
        start_node = Sample(start_point.x, start_point.y)
        goal_coord = Coordinate(goal_point.x, goal_point.y)
    
        if self.sample_inside_obstacles(start_node):
            raise ValueError("Starting point is inside an obstacle.")

        self.samples.append(start_node)

        # Main loop
        while len(self.samples) < max_samples:
            
            # Generate raw random coordinates
            rand_x = np.random.uniform(self.x_range_min, self.x_range_max)
            rand_y = np.random.uniform(self.y_range_min, self.y_range_max)
            candidate_coord = Coordinate(rand_x, rand_y)

            # Find the nearest existing sample
            nearest_sample = self.get_nearest_sample(candidate_coord)

            # Steer: Calculate the actual point we want to create
            steered_candidate_coord = self.steer(nearest_sample, candidate_coord, max_steering_dist)

            # Check before creating the Sample object if the steered candidate is valid (not in obstacle and path is clear)
            if self.sample_inside_obstacles(steered_candidate_coord) or \
               self.path_collision_between(nearest_sample, steered_candidate_coord):
                continue

            # Create the Sample
            new_sample = Sample(steered_candidate_coord.x, steered_candidate_coord.y)
            nearest_sample.connect(new_sample)
            self.samples.append(new_sample)

            # Check Goal
            if new_sample.get_distance(goal_coord) < min_distance_to_goal:
                print(f"Goal reached! Path found with {len(self.samples)} nodes.")
                break

        #print(f"Total valid samples in tree: {len(self.samples)}")
        if Sample.num_samples >= max_samples:
            print("Maximum number of samples reached without connecting to the goal, FAILED to find a path.")      

    def rpm_algorithm(self, start_point, goal_point, max_samples=1000):
        self.samples = []
        Sample.num_samples = 0 # Reset counter

        # Add start as the first sample in the tree
        start_node = Sample(start_point.x, start_point.y)
        self.samples.append(start_node)
        
        # Sample random points in the C-space and add them to the tree
        while len(self.samples) < max_samples-1:
            rand_x = np.random.uniform(self.x_range_min, self.x_range_max)
            rand_y = np.random.uniform(self.y_range_min, self.y_range_max)
            random_coord = Coordinate(rand_x, rand_y)

            # Only add if the coordinate is not inside an obstacle
            if not self.sample_inside_obstacles(random_coord):
                new_sample = Sample(random_coord.x, random_coord.y)
                self.samples.append(new_sample)

        # Add end as the last sample in the tree
        end_node = Sample(goal_point.x, goal_point.y)
        self.samples.append(end_node)

        # Connect each sample to its k-nearest neighbors
        for sample in self.samples:
            # Get the kth closest neighbors
            k_nearest_samples = self.get_kth_nearest_sample(sample, k=8)
            
            for near_sample in k_nearest_samples:
                # Avoid connecting a node to itself
                if sample.get_id() == near_sample.get_id():
                    continue
                
                # Connect if path is clear and they aren't already connected
                if self.path_collision_between(sample, near_sample) == False:
                    if sample.is_connected(near_sample) == False:
                        sample.connect(near_sample)

        print(f"RPM Roadmap constructed with {len(self.samples)} nodes.")


   
class Coordinate:
    """
    Coordinate class represents a point in the 2D space with x and y coordinates. 
    It also provides a method to calculate the distance to another coordinate. 
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_coordinates(self):
        return (self.x, self.y)
    
    def get_distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)




class Sample(Coordinate):
    """
    Sample class represents a point in the C-space that can be connected to other 
    samples to form the RRT tree. It inherits from the Coordinate class and adds functionality for 
    connecting to other samples and keeping track of the number of samples generated.
    """
    
    num_samples = 0
    
    def __init__(self, x, y):
        
        Sample.num_samples += 1
        self.id = Sample.num_samples
    

        super().__init__(x, y)

        self.connected = []

    
    def get_id(self):
        return self.id
    
    def connect(self, other_sample):
        self.connected.append(other_sample)
        other_sample.connected.append(self)
        print(f"Connected Sample {self.id} at ({self.x}, {self.y}) to Sample {other_sample.id} at ({other_sample.x}, {other_sample.y}).")

    def is_connected(self, other_sample):
        return other_sample in self.connected




class Obstacle(Coordinate):
    """
    Obstacle class represents a circular obstacle in the C-space. 
    It inherits from the Coordinate class and adds a radius attribute to define the size of the obstacle. 
    The constructor also assigns a unique ID to each obstacle for easier debugging and visualization.
    """
    num_obstacles = 0

    def __init__(self, center, radius):
        
        Obstacle.num_obstacles += 1
        
        super().__init__(center[0], center[1])
        
        self.radius = radius
        self.obstacle_id = self.num_obstacles
        
        print(f"Created Obstacle {self.obstacle_id}: Center=({self.x}, {self.y}), Radius={self.radius}")
    
    
class Node(Sample):
    """
    Node class represents a node in the graph used for path planning. 
    It inherits from the Sample class and adds attributes for heuristic value, past cost, estimated total
    cost, parent node, and neighbors. This class is used in the A* algorithm to keep track of the path and costs.
    """
    def __init__(self, id, x, y, h):
        
        super().__init__(x, y)
        
        self.node_id = id
        self.hvalue = h
        
        self.visited = False
        self.past_cost = float('inf')
        self.estimated_total_cost = float('inf')
        self.parent = None

        self.Neighbors = []
    
    
    def add_neighbor(self, neighbor_node):
        self.Neighbors.append(neighbor_node)
    
    def get_neighbors(self):
        return self.Neighbors
    
    def get_id(self):
        return self.node_id
    
    def get_coordinates(self):
        return (self.x, self.y)
    
    def set_hvalue(self, hvalue):
        self.hvalue = hvalue
    
    def get_hvalue(self):
        return self.hvalue
    
    def set_visited(self, visited):
        self.visited = visited

    def is_visited(self):
        return self.visited
    
    def set_past_cost(self, tentative_cost):
        self.past_cost = tentative_cost

    def get_past_cost(self):
        return self.past_cost
    
    def set_estimated_total_cost(self, estimated_total_cost):
        self.estimated_total_cost = estimated_total_cost

    def get_estimated_total_cost(self):
        return self.estimated_total_cost
    
    def set_parent(self, parent_node):
        self.parent = parent_node

    def get_parent(self):
        return self.parent
