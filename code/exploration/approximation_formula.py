import numpy as np

p=9.62

for N in range(1,16):
    I= (1/2 )* N *( np.log2(1 + (p**2/ N)))
    print(f"N={N:2d} => I = {I:.4f} bits")
