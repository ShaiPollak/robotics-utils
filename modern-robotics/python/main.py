import os
from graph_manager import GraphManager
import search_algorithms as Search_Algorithms
from space import Tree, Sample, Coordinate, Node



def main():
##################################################
    # FILE PATHS #
    
    base_path = os.getcwd()
    obstacle_csv = os.path.join(base_path, 'data/obstacles.csv')
    nodes_csv = os.path.join(base_path, 'data/nodes.csv')
    edges_csv = os.path.join(base_path, 'data/edges.csv')
    path_csv = os.path.join(base_path, 'data/path.csv')

##################################################
    # IMPORTANT CONSTANTS #
    X_RANGE = (-0.5, 0.5)
    Y_RANGE = (-0.5, 0.5)
    
    OBJECT_RADIUS = 0.01
    MAX_NUM_SAMPLES = 1000

    MIN_DISTANCE_TO_GOAL = 0.03
    MAX_STEERING_DIST = 0.05

    start_point = Coordinate(-0.4, -0.4)
    goal_point = Coordinate(0.4, 0.4)

    SAMPLING_METHOD = 'RRT' # 'RRT' or 'RPM'

##################################################
      
    # Graph Initialization
    graph_manager = GraphManager()
    
    graph_manager.set_nodes_file_path(nodes_csv)
    graph_manager.set_edges_file_path(edges_csv)
    graph_manager.set_path_file_path(path_csv)
    graph_manager.set_obstacles_file_path(obstacle_csv)

    # Load obstacles from CSV and generate obstacle objects
    obstacles = graph_manager.generate_obstacles()

    #Initialize the RRT tree and generate samples in the C-space
    tree = Tree(x_range=X_RANGE, y_range=Y_RANGE, obstacles=obstacles, object_radius=OBJECT_RADIUS)
    tree.generate_tree(start_point, goal_point, num_samples=MAX_NUM_SAMPLES, method=SAMPLING_METHOD, min_distance_to_goal=MIN_DISTANCE_TO_GOAL)

    # Get the generated samples from the tree and save them to CSV files
    samples = tree.get_samples()
    print(f"Generated {len(samples)} samples in the C-space using {SAMPLING_METHOD} algorithm.")
    
    # Now save the generated samples to a CSV file for visualization or further processing
    graph_manager.save_nodes_to_csv(samples, goal_point)
    graph_manager.save_edges_to_csv(samples)

    print("Nodes and edges have been saved to CSV files.")
     
    # Load graph from CSV files
    nodes, distances_matrix = graph_manager.load_graph()
    print(f"Loaded {len(nodes)} nodes and edge distances from CSV files.")

    # Define start and goal nodes (using IDs from nodes.csv)
    try:
        start_node = nodes[1]
        max_node_id = max(nodes.keys())
        goal_node = nodes[max_node_id] #SET IT MANUALLY TO THE GOAL NODE ID YOU WANT
    except KeyError as e:
        print(f"Error: Node ID {e} not found in nodes.csv")
        return

    print(f"Searching path from Node {start_node.get_id()} to Node {goal_node.get_id()}...")

    # Initialize and run A* algorithm
    path = Search_Algorithms.a_star_algorithm(start_node, goal_node, distances_matrix)

    # Save and print the results
    if path:
        print("\n--- Path Found! ---")
        for node in path:
            print(f"Step: Node {node.get_id()} | Coord: {node.get_coordinates()}")
            graph_manager.save_path_to_csv(path)
    else:
        print("\nNo path could be found between the selected nodes.")

if __name__ == "__main__":
    main()

