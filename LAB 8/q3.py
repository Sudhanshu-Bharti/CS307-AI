# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import sys

# Problem Parameters
max_bikes = 20
discount_rate = 0.9
reward_on_credit = 10
reward_on_moving = -2
reward_on_using_second_parking_lot = -4

class PoissonDistribution:
    def __init__(self, lmda):  # Fixed method name from _init_ to __init__
        self.lmda = lmda
        epsilon = 0.01
        self.alpha = 0
        self.vals = {}
        summer = 0
        state = 1

        while True:
            if state == 1:
                temp = poisson.pmf(self.alpha, self.lmda)
                if temp <= epsilon:
                    self.alpha += 1
                else:
                    self.vals[self.alpha] = temp
                    summer += temp
                    self.beta = self.alpha + 1
                    state = 2
            elif state == 2:
                temp = poisson.pmf(self.beta, self.lmda)
                if temp > epsilon:
                    self.vals[self.beta] = temp
                    summer += temp
                    self.beta += 1
                else:
                    break

        added_val = (1 - summer) / (self.beta - self.alpha)
        for key in self.vals:
            self.vals[key] += added_val

    def f(self, n):
        return self.vals.get(n, 0)

class Location:
    def __init__(self, req, ret):
        self.alpha = req
        self.beta = ret
        self.poisson_alpha = PoissonDistribution(self.alpha)
        self.poisson_beta = PoissonDistribution(self.beta)

# Location initialization
A = Location(3, 3)
B = Location(4, 2)

# Initialize value and policy matrices
value = np.zeros((max_bikes + 1, max_bikes + 1))
policy = value.copy().astype(int)

def apply_action(state, action):
    return [
        max(min(state[0] - action, max_bikes), 0),
        max(min(state[1] + action, max_bikes), 0)
    ]

def expected_reward(state, action):
    global value
    si = 0
    new_state = apply_action(state, action)

    if action <= 0:
        si += reward_on_moving * abs(action)  
    else:
        si += reward_on_moving * action 

    if new_state[0] > 10:
        si += reward_on_using_second_parking_lot
    if new_state[1] > 10:
        si += reward_on_using_second_parking_lot

    for Aalpha in range(A.poisson_alpha.alpha, A.poisson_alpha.beta):
        for Balpha in range(B.poisson_alpha.alpha, B.poisson_alpha.beta):
            for Abeta in range(A.poisson_beta.alpha, A.poisson_beta.beta):
                for Bbeta in range(B.poisson_beta.alpha, B.poisson_beta.beta):
                    p = (
                        A.poisson_alpha.vals[Aalpha]
                        * B.poisson_alpha.vals[Balpha]
                        * A.poisson_beta.vals[Abeta]
                        * B.poisson_beta.vals[Bbeta]
                    )

                    valid_requests_A = min(new_state[0], Aalpha)
                    valid_requests_B = min(new_state[1], Balpha)
                    reward = (valid_requests_A + valid_requests_B) * reward_on_credit

                    new_s = [
                        max(min(new_state[0] - valid_requests_A + Abeta, max_bikes), 0),
                        max(min(new_state[1] - valid_requests_B + Bbeta, max_bikes), 0)
                    ]

                    si += p * (reward + discount_rate * value[new_s[0]][new_s[1]])

    return si

def policy_evaluation():
    global value
    e = policy_evaluation.e
    policy_evaluation.e /= 10

    while True:
        delta = 0
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                old_val = value[i][j]
                value[i][j] = expected_reward([i, j], policy[i][j])
                delta = max(delta, abs(value[i][j] - old_val))
                print('.', end='', flush=True)
        print(delta, flush=True)

        if delta < e:
            break

policy_evaluation.e = 50

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

# Main loop
while True:
    policy_evaluation()
    stable = policy_improvement()
    save_value()
    save_policy()
    if stable:
        break