from string import ascii_lowercase
import random
from itertools import combinations
import numpy as np

def creatproblem(n,k,m):
        positive_var = list(ascii_lowercase)[:n]
        negative_var = [var.upper() for var in positive_var]
        variables = positive_var + negative_var
        problem = []
        threshold = 10       
        i = 0
        comb = list(combinations(variables,k))
        
        while i<threshold:
            c = random.sample(comb,m)
            if c not in problem:
                problem.append(c)
                i += 1
        
        problems_new = []
        for c in problem:
            temp = []
            temp = [list(sub) for sub in c]
            problems_new.append(temp)
        return  variables,problems_new 

def random_assign(variables,n):
    litral = list(np.random.choice(2,n))
    negation = [abs(i-1) for i in litral]
    assign = litral + negation
    return dict(zip(variables,assign))

n = 5
k = 3
m = 3
var,prob = creatproblem(n,k,m)
print(var)
for i in prob:
    print(i)

print(random_assign(var,n))