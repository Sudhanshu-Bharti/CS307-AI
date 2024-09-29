from collections import deque
import random

# Class to represent a node in the state space
class Node:
    def __init__(self, state, parent=None):
        self.state = state  # Current puzzle configuration
        self.parent = parent  # Reference to the parent node

# Function to generate the next possible states from the current node
def get_successors(node):
    successors = []  # List to hold all valid successor states
    index = node.state.index(0)  # Find the position of the blank tile (represented by 0)
    
    # Determine row and column of the blank tile
    row = index // 3
    col = index % 3
    
    # Define possible moves based on the current row
    if row == 0:
        moves = [3]  # Can only move down
    elif row == 1:
        moves = [-3, 3]  # Can move up or down
    else:
        moves = [-3]  # Can only move up

    # Define possible moves based on the current column
    if col == 0:
        moves += [1]  # Can move right
    elif col == 1:
        moves += [-1, 1]  # Can move left or right
    else:
        moves += [-1]  # Can move left

    # Generate new states by applying valid moves
    for move in moves:
        new_index = index + move
        if 0 <= new_index < 9:  # Ensure the move stays within bounds
            new_state = list(node.state)  # Create a copy of the current state
            # Swap the blank tile with the tile at the new index
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            successors.append(Node(new_state, node))  # Add the resulting state to the successors list
    return successors

# Breadth-First Search (BFS) algorithm to solve the Puzzle-8 problem
def bfs(start_state, goal_state):
    start_node = Node(start_state)  # Initialize the starting node
    goal_node = Node(goal_state)  # Goal state node
    queue = deque([start_node])  # Queue for BFS exploration
    visited = set()  # Set to track visited states
    nodes_explored = 0  # Counter for explored nodes
    
    while queue:
        node = queue.popleft()  # Pop the first node from the queue
        if tuple(node.state) in visited:
            continue  # Skip already visited states
        visited.add(tuple(node.state))  # Mark the state as visited
        nodes_explored += 1  # Increment the counter for explored nodes
        
        # If goal state is found, reconstruct the solution path
        if node.state == list(goal_node.state):
            path = []
            while node:
                path.append(node.state)  # Backtrack to collect the path
                node = node.parent
            print('Total nodes explored:', nodes_explored)
            return path[::-1]  # Return the path from start to goal
        
        # Expand the current node's successors
        for successor in get_successors(node):
            queue.append(successor)  # Add successors to the queue for exploration
    
    return None  # Return None if no solution is found

# Initial state configuration for the Puzzle-8
start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
s_node = Node(start_state)

# Depth at which we want to generate a goal state
D = 20
d = 0

# Generate a random goal state by making D moves from the initial state
while d <= D:
    goal_state = random.choice(list(get_successors(s_node))).state
    s_node = Node(goal_state)
    d += 1

# Solve the Puzzle-8 problem using BFS
solution = bfs(start_state, goal_state)

# Print the solution steps if a solution is found
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
