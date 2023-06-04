# import os 
# for filename in os.listdir('./'):
#     with open(filename, 'r') as f:
#         print(filename, '= [', end='')
#         for line in f.readlines():
#             if 'At round 1000 benign test accu' in line:
#                 print(float(line.split()[-1]), ', ', end='')
#         print(']')

import matplotlib.pyplot as plt

fmnist_a1 = [ ]
fmnist_b1 = [ ]

fmnist_a2 = [ ]
fmnist_b2 = [ ]

fmnist_a3 = [ ]
fmnist_b3 = [ ]
