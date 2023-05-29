import numpy as np
import random
import math
from scipy.stats import multivariate_normal

EPS=1e-9
class GMM:
    def __init__(self, params):
        self.params = params

    def run(self, X):
        n= X.shape[0]
        m= X.shape[1]
        k= self.params['k']

        # print("Start ",k)
        
        # initialize
        phi = np.zeros(k)
        for i in range(0,k):
            phi[i]=(1/k)

        took=random.sample(range(0,n),k)
        sigma=np.zeros((k,m,m))
        mu=np.zeros((k,m))
        for j in range(0,k):
            mu[j]=X[took[j]]
            sigma[j]=np.identity(m)

        last=-1e9
        likelihood=-1e9
        cnt=0
        for step in range(0,self.params['iter']):
            
            # E-step
            P=np.zeros((n, k))
            for j in range(0,k):
                temp=multivariate_normal.pdf(mean=mu[j], cov=sigma[j], x=X,allow_singular=True)
                temp=np.expand_dims(temp, axis=1)
                P[:,j]=(phi[j]*temp).flatten()

            row_sum=P.sum(axis=1)
            P=P/row_sum[:,None]

            # log likelihood
            likelihood=np.log(row_sum).sum()
            if abs(likelihood-last)<EPS:
                cnt+=1
            else:
                cnt=0
            if cnt==5:
                break
            last=likelihood

            # M-step
            for j in range(0,k):
                column=P[:,j]
                tot=column.sum()
                column=np.expand_dims(column, axis=1)
                phi[j]=tot/n
                mu[j]=np.sum(column.T.dot(X),axis=0)/tot
                sigma[j] = (column * (X - mu[j])).T.dot(X - mu[j]) / tot

        return likelihood




            
