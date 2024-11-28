import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import sys

# Problem Parameters
class jcp:
    @staticmethod
    def max_bikes():
        return 20
    
    @staticmethod
    def γ():
        return 0.9
    
    @staticmethod
    def credit_reward():
        return 10
    
    @staticmethod
    def moving_reward():
        return -2

class poisson_:
    def __init__(self, λ):  # Fixed constructor method name
        self.λ = λ
        ε = 0.01
        self.α = 0
        state = 1
        self.vals = {}
        summer = 0
        
        while True:
            if state == 1:
                temp = poisson.pmf(self.α, self.λ) 
                if temp <= ε:
                    self.α += 1
                else:
                    self.vals[self.α] = temp
                    summer += temp
                    self.β = self.α + 1
                    state = 2
            elif state == 2:
                temp = poisson.pmf(self.β, self.λ)
                if temp > ε:
                    self.vals[self.β] = temp
                    summer += temp
                    self.β += 1
                else:
                    break    
        
        added_val = (1 - summer) / (self.β - self.α)
        for key in self.vals:
            self.vals[key] += added_val
           
    def f(self, n):
        return self.vals.get(n, 0)

# A class holding the properties of a location together
class location:
    def __init__(self, req, ret):
        self.α = req
        self.β = ret
        self.poissonα = poisson_(self.α)  # Corrected initialization
        self.poissonβ = poisson_(self.β)  # Corrected initialization

# Location initialization
A = location(3, 3)
B = location(4, 2)

# Initializing the value and policy matrices
value = np.zeros((jcp.max_bikes() + 1, jcp.max_bikes() + 1))
policy = value.copy().astype(int)

def expected_reward(state, action):
    global value
    ψ = 0
    new_state = [
        max(min(state[0] - action, jcp.max_bikes()), 0),
        max(min(state[1] + action, jcp.max_bikes()), 0),
    ]
    ψ += jcp.moving_reward() * abs(action)
    
    for Aα in range(A.poissonα.α, A.poissonα.β):
        for Bα in range(B.poissonα.α, B.poissonα.β):
            for Aβ in range(A.poissonβ.α, A.poissonβ.β):
                for Bβ in range(B.poissonβ.α, B.poissonβ.β):
                    ζ = (
                        A.poissonα.vals[Aα] *
                        B.poissonα.vals[Bα] *
                        A.poissonβ.vals[Aβ] *
                        B.poissonβ.vals[Bβ]
                    )
                    
                    valid_requests_A = min(new_state[0], Aα)
                    valid_requests_B = min(new_state[1], Bα)
                    
                    rew = (valid_requests_A + valid_requests_B) * jcp.credit_reward()
                    
                    new_s = [
                        max(min(new_state[0] - valid_requests_A + Aβ, jcp.max_bikes()), 0),
                        max(min(new_state[1] - valid_requests_B + Bβ, jcp.max_bikes()), 0),
                    ]
                    
                    ψ += ζ * (rew + jcp.γ() * value[new_s[0]][new_s[1]])
                    
    return ψ

def policy_evaluation():
    global value
    ε = policy_evaluation.ε
    policy_evaluation.ε /= 10 
    
    while True:
        δ = 0
        
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                old_val = value[i][j]
                value[i][j] = expected_reward([i, j], policy[i][j])
                δ = max(δ, abs(value[i][j] - old_val))
                
                print('.', end='')
                sys.stdout.flush()
        print(δ)
        sys.stdout.flush()
    
        if δ < ε:
            break

policy_evaluation.ε = 50

def policy_improvement():
    global policy
    policy_stable = True
    
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            old_action = policy[i][j]
            max_act_val = None
            max_act = None
            
            τ12 = min(i, 5)
            τ21 = -min(j, 5)
            
            for act in range(τ21, τ12 + 1):
                σ = expected_reward([i, j], act)
                if max_act_val is None or max_act_val < σ:
                    max_act_val = σ
                    max_act = act
                
            policy[i][j] = max_act
            
            if old_action != policy[i][j]:
                policy_stable = False
    
    return policy_stable

def save_policy():
    save_policy.counter += 1
    ax = sns.heatmap(policy, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig(f'policy{save_policy.counter}.svg')
    plt.close()

def save_value():
    save_value.counter += 1
    ax = sns.heatmap(value, linewidth=0.5)
    ax.invert_yaxis()
    plt.savefig(f'value{save_value.counter}.svg')
    plt.close()

save_policy.counter = 0
save_value.counter = 0

while True:
    policy_evaluation()
    ρ = policy_improvement()
    save_value()
    save_policy()
    if ρ:
        break