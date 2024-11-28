import numpy as np


wall = [(1,1)]

possible_actions = ['L','R','U','D']
terminate_states = ((1,3),(2,3))


# non - deterministic action
prob_actions = {'L':0.25,'R':0.25,'U':0.25,'D':0.25}

# environment action corresponding to Agent
environment_left = {'L':'D','R':'U','U':'L','D':'R'}
environment_right = {'L':'U','R':'D','U':'R','D':'L'}

def is_valid(i,j):
    return (i,j) not in wall and i >= 0 and i < 3 and j >= 0 and j < 4

def print_values(V):
  for i in range(2,-1,-1):
    print("--- --- --- --- --- --- ---")
    for j in range(4):
      v = V[i][j]
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")

def func(action,i,j):
    if action == 'L':
        new_state = (i,j-1)
    elif action == 'R':
        new_state = (i,j+1)
    elif action == 'U':
        new_state = (i+1,j)
    else:
        new_state = (i-1,j)   

    return new_state

def find_value_function(i,j,reward,reward_matrix,tolerance_rate=1):
    value = 0
    for action in possible_actions:
        # desired action with 0.8 probability
        state_x,state_y = func(action,i,j)
        if is_valid(state_x,state_y):
            desired_action_value = (reward_matrix[state_x][state_y] + tolerance_rate*V_pie[state_x][state_y])
        else:
            desired_action_value = (reward_matrix[i][j] + tolerance_rate*V_pie[i][j])
        
        # environment action with 0.1 probability
        state_x,state_y = func(environment_left[action],i,j)
        if is_valid(state_x,state_y):
            env_action_left_value = (reward_matrix[state_x][state_y] + tolerance_rate*V_pie[state_x][state_y])
        else:
            env_action_left_value = (reward_matrix[i][j] + tolerance_rate*V_pie[i][j])
        
        # environment action with 0.1 probability 
        state_x,state_y = func(environment_right[action],i,j)
        if is_valid(state_x,state_y):
            env_action_right_value = (reward_matrix[state_x][state_y] + tolerance_rate*V_pie[state_x][state_y])
        else:
            env_action_right_value = (reward_matrix[i][j] + tolerance_rate*V_pie[i][j])
        
        value_to_action = desired_action_value*0.8+env_action_left_value*0.1+env_action_right_value*0.1        

        value += value_to_action*prob_actions[action]

    return value

# iterative policy evaluation
def iterative_policy_evaluation(iter,theta,reward,reward_matrix,V_pie):
    while True:
        delta = 0
        for i in range(3):
            for j in range(4):
                state = (i,j)
                if state in terminate_states or state in wall:
                    continue
                v = V_pie[i][j]
                V_pie[i][j] = find_value_function(i,j,reward,reward_matrix)
                delta = max(delta,abs(v-V_pie[i][j]))
        iter += 1
        if delta < theta:
            print(f"Total Iterations :{iter}")
            break 
    print_values(V_pie)

def update_reward_matrix(reward):
    reward_matrix = [[reward for _ in range(4)] for _ in range(3)]
    reward_matrix[2][3] = 1
    reward_matrix[1][3] = -1
    return reward_matrix
    
def initialize_V_pie():
    V_pie = [[0 for _ in range(4)]for _ in range(3)]
    return V_pie    

rewards = [-0.04,-2,0.1,0.02,1]

if __name__ == "__main__":
    for reward in rewards:
        print(f"At r(S) : {reward}")
        reward_matrix = update_reward_matrix(reward)
        V_pie = initialize_V_pie()
        iterative_policy_evaluation(0,1e-7,reward,reward_matrix,V_pie)
        print("\n**************\n")