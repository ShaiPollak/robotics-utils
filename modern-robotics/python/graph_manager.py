
import csv
import os
from space import Obstacle, Node

class GraphManager:
    def __init__(self):
        self.obstacles_path = None
        self.nodes_path = None
        self.edges_path = None
        self.path_path = None
        self.nodes = {}
        self.edge_matrix = []

    
    def set_nodes_file_path(self, path):
        self.nodes_path = path

    def set_edges_file_path(self, path):
        self.edges_path = path

    def set_path_file_path(self, path):
        self.path_path = path

    def set_edges_file_path(self, path):
        self.edges_path = path

    def set_obstacles_file_path(self, path):
        self.obstacles_path = path

    def generate_obstacles(self):
        obstacles = []
        if self.obstacles_path is None:
            print("Obstacles path not set.")
            return obstacles
        
        with open(self.obstacles_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#'): continue
                center_x, center_y, radius = float(row[0]), float(row[1]), float(row[2])/2
                obstacles.append(Obstacle((center_x, center_y), radius))
        return obstacles
    
    
    def load_graph(self):
        # Load nodes and edges from CSV files and populate the graph structure
        self.nodes = self._load_nodes()
        self.edge_matrix = self._load_edges()
        return self.nodes, self.edge_matrix

    def _load_nodes(self):
        nodes = {}
        with open(self.nodes_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#'): continue
                n_id, x, y, h = int(row[0]), float(row[1]), float(row[2]), float(row[3])
                nodes[n_id] = Node(n_id, x, y, h)
        return nodes

    def _load_edges(self):
        num_nodes = max(self.nodes.keys())
        # Initialize edge matrix with zeros
        matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        
        with open(self.edges_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith('#'): continue
                id1, id2, cost = int(row[0]), int(row[1]), float(row[2])
                
                # update neighbors in Node objects (bidirectional)
                node1, node2 = self.nodes[id1], self.nodes[id2]
                if node2 not in node1.get_neighbors():
                    node1.add_neighbor(node2)
                if node1 not in node2.get_neighbors():
                    node2.add_neighbor(node1)
                
                # Update edge cost in the matrix (bidirectional)
                matrix[id1-1][id2-1] = cost
                matrix[id2-1][id1-1] = cost
        return matrix
    

    def save_edges_to_csv(self, samples):
        with open(self.edges_path, 'w') as f:
            # Header row
            f.write("# NodeId, NeighborId, Distance\n")
            
            # Set to track already written edges to avoid duplicates (e.g., 1-2 and 2-1)
            written_edges = set()
            
            for sample in samples:
                for connection in sample.connected:
                    # Create a unique edge identifier by sorting IDs
                    # This ensures (1, 2) and (2, 1) are treated as the same edge
                    edge = tuple(sorted((sample.get_id(), connection.get_id())))
                    
                    if edge not in written_edges:
                        # Calculate distance
                        dist = sample.get_distance(connection)
                        
                        # Write to file with 3 decimal places using :.3f
                        f.write(f"{sample.get_id()}, {connection.get_id()}, {dist:.3f}\n")
                        
                        # Mark this edge as written
                        written_edges.add(edge)


    def save_nodes_to_csv(self, samples, goal_point):
        with open(self.nodes_path, 'w') as f:
            f.write("# NodeID, X, Y, H\n")
            for sample in samples:
                #POLYMORPHIZM FOR THE WIN!
                h_value = sample.get_distance(goal_point)
                f.write(f"{sample.get_id()}, {sample.x:.3f}, {sample.y:.3f}, {h_value:.3f}\n")

    def save_path_to_csv(self, path):
        with open(self.path_path, 'w') as f:
            f.write("# NodeIDs:\n")
            for node in path:
                f.write(f"{node.get_id()},")
            
            # Clear last comma and add newline
            f.seek(f.tell() - 1)
            f.write("\n")           
