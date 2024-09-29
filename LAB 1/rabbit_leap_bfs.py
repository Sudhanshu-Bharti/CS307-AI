from collections import deque

def print_state(state):
    """Function to print the current state of the stones."""
    print(" ".join(str(x) for x in state))


def is_goal_state(state):
    """Check if the current state is the goal state."""
    return state == [-1, -1, -1, 0, 1, 1, 1]


def get_successor(state):
    """Find all possible moves for the rabbits."""
    moves = []
    for i in range(len(state)):
        if state[i] == 1:  # East-bound rabbit
            # Move one step forward if possible
            if i + 1 < len(state) and state[i + 1] == 0:
                moves.append((i, i + 1))
            # Jump over one rabbit if possible
            if i + 2 < len(state) and state[i + 1] == -1 and state[i + 2] == 0:
                moves.append((i, i + 2))
        elif state[i] == -1:  # West-bound rabbit
            # Move one step forward if possible
            if i - 1 >= 0 and state[i - 1] == 0:
                moves.append((i, i - 1))
            # Jump over one rabbit if possible
            if i - 2 >= 0 and state[i - 1] == 1 and state[i - 2] == 0:
                moves.append((i, i - 2))
    return moves


def make_move(state, move):
    """Apply a move to the state."""
    new_state = state[:]
    new_state[move[1]] = new_state[move[0]]
    new_state[move[0]] = 0
    return new_state


# BFS Implementation
def bfs(initial_state):
    """Solve the Rabbit Leap problem using BFS for optimal solution."""
    queue = deque([(initial_state, [])])  # (current state, path of moves)
    visited = set()

    total_nodes_explored = 0
    max_queue_size = 1

    while queue:
        (current_state, path) = queue.popleft()
        total_nodes_explored += 1

        # Track maximum size of the queue
        max_queue_size = max(max_queue_size, len(queue))

        # Convert current_state to a tuple for the visited set
        state_tuple = tuple(current_state)

        # Check if we have already visited this state
        if state_tuple in visited:
            continue

        visited.add(state_tuple)
        path = path + [current_state]

        # Check if we've reached the goal state
        if is_goal_state(current_state):
            print(f"Total Nodes Explored (BFS): {total_nodes_explored}")
            print(f"Max Queue Size (BFS): {max_queue_size}")
            return path

        # Find all possible moves from the current state
        moves = get_successor(current_state)
        for move in moves:
            new_state = make_move(list(current_state), move)
            queue.append((new_state, path))

    return None  # If no solution is found




# Comparison Function
def compare_solutions(initial_state):
    # BFS
    bfs_path = bfs(initial_state)
    if bfs_path:
        print("\nBFS Solution Found in {} steps:".format(len(bfs_path) - 1))
        for step in bfs_path:
            print(step)
        print(f"Total Nodes in Solution (BFS): {len(bfs_path)}")
    else:
        print("No solution found using BFS.")

    

# Main
initial_state = [1, 1, 1, 0, -1, -1, -1]
compare_solutions(initial_state)
