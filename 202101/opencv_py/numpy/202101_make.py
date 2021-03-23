import numpy as np
from numpy import linalg as LA
import scipy as sp

a = np.random.randint(1 ,10 ,10);
print(a)

np.random.shuffle(a);
print(a)

a = np.array([[-2 ,-36 ,0] ,[-36 ,-23 ,0] ,[0 ,0 ,3]])
w,v = LA.eig(a)

print(w)
print(v)
print("I ate noodles")
