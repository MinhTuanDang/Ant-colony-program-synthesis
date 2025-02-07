import random
import math

# --- Utility functions and definitions ---

def protected_div(a, b):
    """Protected division to avoid division by zero."""
    try:
        if abs(b) < 1e-6:
            return 1.0
        return a / b
    except Exception:
        return 1.0

# Define the available functions and map them to Python operations.
FUNCTIONS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': protected_div,
}
FUNCTION_NAMES = list(FUNCTIONS.keys())

# Define the terminal set.
TERMINALS = ['x', 1.0, 2.0, 5.0]

# --- Data structures ---

class Node:
    def __init__(self, node_id, node_type, value):
        """
        node_type: 'function' or 'terminal'
        value: for function nodes, the operator as a string (e.g., '+'); for terminal nodes, either 'x' or a constant.
        """
        self.id = node_id
        self.type = node_type
        self.value = value
        # For functions in our examples, we assume arity = 2.
        self.arity = 2 if node_type == 'function' else 0

    def __repr__(self):
        return f"Node(id={self.id}, type={self.type}, value={self.value})"


class Graph:
    def __init__(self, nodes, edge_prob=0.6):
        """
        Create a graph with a dictionary of nodes (keyed by id) and randomly generated undirected edges.
        Each edge is represented as a tuple (min_id, max_id) mapped to a pheromone level.
        """
        # Store nodes in a dict for fast access.
        self.nodes = {node.id: node for node in nodes}
        self.edges = {}  # key: (min_id, max_id), value: pheromone level
        self.generate_edges(edge_prob)
    
    def generate_edges(self, prob):
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < prob:
                    key = (node_ids[i], node_ids[j])
                    self.edges[key] = 1.0  # initialize pheromone level
    
    def get_neighbors(self, node_id):
        """
        Return a list of neighbors of node_id as tuples (neighbor_id, pheromone, edge_key).
        """
        neighbors = []
        for (i, j), pheromone in self.edges.items():
            if i == node_id:
                neighbors.append((j, pheromone, (i, j)))
            elif j == node_id:
                neighbors.append((i, pheromone, (i, j)))
        return neighbors

    def update_pheromone(self, edge_key, amount):
        """Increase the pheromone level on the given edge."""
        if edge_key in self.edges:
            self.edges[edge_key] += amount

    def evaporate_pheromones(self):
        """Evaporate pheromone on every edge by a random factor between 0% and 25%."""
        for key in self.edges:
            evaporation_rate = random.uniform(0, 0.25)
            self.edges[key] *= (1 - evaporation_rate)

    def __repr__(self):
        return f"Graph(nodes={len(self.nodes)}, edges={len(self.edges)})"


# --- Ant Traversal and Program Tree Construction ---

def ant_traverse(graph, current_node_id, depth, min_depth=3, max_depth=8, learning_rate=0.95):
    """
    Recursively traverse the graph starting from the current_node.
    Returns a tuple:
      - tree: a representation of the program tree (for a function node, a tuple (operator, left, right); 
              for a terminal, just its value).
      - edges_used: a set of edges (represented by their key) used in this traversal.
    
    The traversal is biased by pheromone levels and constrained by minimum/maximum depth:
      - If depth < min_depth, force selection of a function node if one is available.
      - If depth > max_depth, force selection of a terminal node if available.
    """
    current_node = graph.nodes[current_node_id]
    edges_used = set()
    
    if current_node.type == 'terminal':
        # Reached a terminal: this is a leaf of the program tree.
        return current_node.value, edges_used

    # The current node is a function node.
    children = []
    for param in range(current_node.arity):
        neighbors = graph.get_neighbors(current_node_id)
        
        # Enforce depth constraints.
        forced_type = None
        if depth < min_depth:
            if any(graph.nodes[nb].type == 'function' for nb, _, _ in neighbors):
                forced_type = 'function'
        elif depth > max_depth:
            if any(graph.nodes[nb].type == 'terminal' for nb, _, _ in neighbors):
                forced_type = 'terminal'
        
        # Filter candidates based on forced type.
        if forced_type:
            candidates = [(nb, pheromone, edge_key)
                          for nb, pheromone, edge_key in neighbors
                          if graph.nodes[nb].type == forced_type]
            if not candidates:
                candidates = neighbors
        else:
            candidates = neighbors

        if not candidates:
            # If no neighbor is available, return a default terminal.
            child_tree = 1.0
            children.append(child_tree)
            continue
        
        # Decide on next node using either roulette wheel selection (pheromone-biased) or uniform random.
        if random.random() < learning_rate:
            total_pheromone = sum(pheromone for _, pheromone, _ in candidates)
            r = random.uniform(0, total_pheromone)
            cumulative = 0.0
            selected = None
            for nb, pheromone, edge_key in candidates:
                cumulative += pheromone
                if cumulative >= r:
                    selected = (nb, edge_key)
                    break
            if selected is None:
                selected = (candidates[-1][0], candidates[-1][2])
        else:
            sel = random.choice(candidates)
            selected = (sel[0], sel[2])
        
        child_node_id, edge_key = selected
        edges_used.add(edge_key)
        # Recursively traverse from the chosen neighbor.
        child_tree, child_edges = ant_traverse(graph, child_node_id, depth + 1,
                                               min_depth, max_depth, learning_rate)
        edges_used.update(child_edges)
        children.append(child_tree)
    
    # Build the program tree for the function node.
    tree = (current_node.value, children[0], children[1])
    return tree, edges_used


# --- Program Evaluation and Fitness Calculation ---

def evaluate_tree(tree, x):
    """
    Evaluate the program tree for a given input x.
    The tree is either a terminal (e.g., 'x' or a constant) or a tuple (operator, left, right).
    """
    if not isinstance(tree, tuple):
        # Terminal: if it is 'x', return the value of x; otherwise, return the constant.
        return x if tree == 'x' else tree
    
    func_name, left, right = tree
    func = FUNCTIONS[func_name]
    a = evaluate_tree(left, x)
    b = evaluate_tree(right, x)
    return func(a, b)

def fitness(tree):
    """
    Compute the fitness of a program tree.
    For a symbolic regression problem, we compare the program output to the expected value.
    In this example the target function is f(x) = 6*x^2 + 10*x + 12.
    The fitness is the sum of absolute differences for x values in the range [-10, 10].
    """
    total_error = 0.0
    for x in range(-10, 11):
        try:
            output = evaluate_tree(tree, x)
        except Exception:
            output = 1e6  # heavy penalty in case of error
        expected = 6 * x**2 + 10 * x + 12
        total_error += abs(output - expected)
    return total_error


# --- Main loop of the Ant Colony Programming algorithm ---

def main():
    random.seed(42)  # For reproducibility

    # Create 80 nodes:
    #   40 function nodes (10 for each: +, -, *, /)
    #   40 terminal nodes (10 for each: 'x', 1.0, 2.0, 5.0)
    nodes = []
    node_id = 0

    # Add function nodes.
    for func in FUNCTION_NAMES:
        for _ in range(10):
            nodes.append(Node(node_id, 'function', func))
            node_id += 1

    # Add terminal nodes.
    for term in TERMINALS:
        for _ in range(10):
            nodes.append(Node(node_id, 'terminal', term))
            node_id += 1

    # Create the graph with the nodes and random edges.
    graph = Graph(nodes, edge_prob=0.6)
    print("Graph created:", graph)

    # Define starting points: choose 10 function nodes (starting points are fixed).
    function_node_ids = [node.id for node in nodes if node.type == 'function']
    starting_points = random.sample(function_node_ids, 10)

    # Parameters for the algorithm.
    generations = 1000
    ants_per_start = 10  # Total ants per generation: 10 starting points * 10 ants = 100 ants.
    best_overall_tree = None
    best_overall_fitness = float('inf')

    # Main evolutionary loop.
    for gen in range(generations):
        ant_solutions = []  # Each element: (program_tree, edges_used, fitness)

        # For each starting point, launch ants_per_start ants.
        for sp in starting_points:
            for _ in range(ants_per_start):
                tree, edges_used = ant_traverse(graph, sp, depth=0,
                                                min_depth=3, max_depth=8, learning_rate=0.95)
                fit = fitness(tree)
                ant_solutions.append((tree, edges_used, fit))

        # Rank the solutions by fitness (lower is better).
        ant_solutions.sort(key=lambda sol: sol[2])
        best_gen_tree, best_gen_edges, best_gen_fit = ant_solutions[0]
        print(f"Generation {gen}: Best fitness = {best_gen_fit}")

        if best_gen_fit < best_overall_fitness:
            best_overall_fitness = best_gen_fit
            best_overall_tree = best_gen_tree

        # --- Global Pheromone Update ---
        # The best 4 ants deposit pheromone along the edges they traversed.
        deposit_amounts = [3.0, 2.0, 1.0, 1.0]
        for i in range(min(4, len(ant_solutions))):
            _, edges_used, _ = ant_solutions[i]
            for edge_key in edges_used:
                graph.update_pheromone(edge_key, deposit_amounts[i])
        
        # --- Pheromone Evaporation ---
        graph.evaporate_pheromones()

    print("Best overall fitness:", best_overall_fitness)
    print("Best program tree:", best_overall_tree)

if __name__ == '__main__':
    main()