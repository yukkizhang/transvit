import sys
import os

DIR = os.path.abspath(__file__)
n = 1
for i in range(n):
    DIR = os.path.dirname(DIR)

sys.path.append(DIR)
