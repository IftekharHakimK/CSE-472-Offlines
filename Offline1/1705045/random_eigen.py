import numpy as np
import random
n=int(input())
while True:
    a=[]
    for i in range(0,n):
        temp=[]
        for j in range(0,n):
            temp.append(random.randint(-10,10))
        a.append(temp)
    M=np.array(a)
    if np.linalg.det(M)!=0:
        break
print("M: ", M)
eigen_values, eigen_vectors = np.linalg.eig(M)
inv_eigen_vectors = np.linalg.inv(eigen_vectors)
D = np.diag(eigen_values)
M_got = eigen_vectors.dot(D).dot(inv_eigen_vectors)
print("M_got: ", M_got)
print(np.allclose(M_got, M))


