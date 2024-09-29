import heapq
import random

# Class to represent each node in the search tree
class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state        # Current puzzle configuration (state)
        self.parent = parent      # Reference to the parent node
        self.g = g                # Cost from the start node (depth or steps taken)
        self.h = h                # Heuristic value (optional, for algorithms like A*)
        self.f = g + h            # Evaluation function f = g + h
    def __lt__(self, other):
        return self.g < other.g   # Less-than method for priority queue (heap)

# Heuristic function placeholder (can be used in A* search)
def heuristic(node, goal_state):
    h = 0
    return h

# Function to generate valid successors (child states) by moving the blank tile (0)
def get_successors(node):
    successors = []  # List to hold valid successors
    index = node.state.index(0)  # Find the position of the blank tile (0)
    
    # Determine row and column based on the index of the blank tile
    row = index // 3
    col = index % 3
    
    # Define valid moves based on the row of the blank tile
    if row == 0:
        moves = [3]  # Can move down
    elif row == 1:
        moves = [-3, 3]  # Can move up or down
    else:
        moves = [-3]  # Can move up
    
    # Add valid column-based moves
    if col == 0:
        moves += [1]  # Can move right
    elif col == 1:
        moves += [-1, 1]  # Can move left or right
    else:
        moves += [-1]  # Can move left

    # Generate new states by performing the valid moves
    for move in moves:
        new_index = index + move
        if 0 <= new_index < 9:  # Ensure the new index is within bounds
            new_state = list(node.state)  # Copy current state
            # Swap the blank tile with the tile in the new position
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            successor = Node(new_state, node, node.g + 1)  # Create a successor node with updated g-value
            print(successor.g)  # Optionally print g-value
            successors.append(successor)  # Add successor to the list
    return successors

# Generic search function using a priority queue (e.g., uniform cost or A*)
def search_agent(start_state, goal_state):
    start_node = Node(start_state)  # Initialize the starting node
    goal_node = Node(goal_state)    # Define the goal state node
    frontier = []  # Priority queue for frontier nodes
    heapq.heappush(frontier, (start_node.g, start_node))  # Add start node to frontier, prioritizing by g-value
    
    visited = set()  # Set to keep track of visited states
    nodes_explored = 0  # Counter for nodes explored

    while frontier:
        _, node = heapq.heappop(frontier)  # Pop node with the lowest g-value from the frontier
        
        if tuple(node.state) in visited:
            continue  # Skip nodes that have already been visited
        
        visited.add(tuple(node.state))  # Mark the current node's state as visited

        # Check if the current node is the goal
        if node.state == goal_node.state:
            path = []
            while node:
                path.append(node.state)  # Reconstruct the path by backtracking from goal to start
                node = node.parent
            return path[::-1]  # Return the path from start to goal
        
        # Expand the current node and add its successors to the frontier
        for successor in get_successors(node):
            heapq.heappush(frontier, (successor.g, successor))  # Add successor to the priority queue
    return None  # Return None if no solution is found

# Initial puzzle state (start state)
start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
s_node = Node(start_state)

# Set the desired depth for generating the goal state
D = 20
d = 0

# Generate a random goal state by applying valid moves to the initial state
while d <= D:
    goal_state = random.choice(get_successors(s_node)).state
    s_node = Node(goal_state)
    d += 1

# Solve the puzzle using the search agent
solution = search_agent(start_state, goal_state)

# Print the solution if found, otherwise print "No solution"
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")