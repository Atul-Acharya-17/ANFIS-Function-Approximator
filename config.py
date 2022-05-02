import numpy as np
import math 


y = []

num_inputs = 2
num_member_funcs = 3

membership_funcs_params = [
    [
        [3, 3, 0],
        [3, 3, 5],
        [3, 3, 10]
    ],
    [
        [0.5, 2, 0],
        [0.5, 2, -1],
        [0.5, 2, 1]
    ]
]

x1 = np.arange(0.0, 10.1, 0.1)
x2 = np.sin(x1)
y = [eval(f'math.cos(2*{x1[i]})/math.exp({x2[i]})') for i in range(x1.shape[0])]