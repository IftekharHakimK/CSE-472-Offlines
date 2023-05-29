import numpy as np
import random
n=int(input())
m=int(input())

a=[]
for i in range(0,n):
    temp=[]
    for j in range(0,m):
        temp.append(random.randint(-10,10))
    a.append(temp)
M=np.array(a)
print("M: ", M)

U, S, Vh = np.linalg.svd(M, full_matrices=True)
print("U: ", U)
print("S: ", S)
print("Vh: ", Vh)

MP_inv = np.linalg.pinv(M)
print("MP_inv: ", MP_inv)

D = np.zeros(M.shape)
for i in range(0,len(S)):
    if S[i]!=0:
        D[i][i]=1/S[i]
D=D.T
MP_inv_2 = Vh.T.dot(D).dot(U.T)
print("MP_inv_2: ", MP_inv_2)

print(np.allclose(MP_inv, MP_inv_2))