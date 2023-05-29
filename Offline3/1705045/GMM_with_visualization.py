import numpy as np
import random
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
EPS=1e-9
class GMM:
    def __init__(self, params):
        self.params = params

    def run(self, X):
        n= X.shape[0]
        m= X.shape[1]
        k= self.params['k']

        plt.ion()
        fig, ax = plt.subplots()
        

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

            ax.clear()
            scatter = ax.scatter(X[:, 0], X[:, 1],c=P.argmax(axis=1))

            X_pca=X
            if m>2:
                pca=PCA(n_components=2)
                pca.fit(X)
                X_pca=pca.transform(X)

            for j in range(0,k):
                m=X.shape[1]
                if m==2:
                    x, y = np.mgrid[min(X[:, 0]):max(X[:, 0]):.01, min(X[:, 1]):max(X[:, 1]):.01]
                    pos = np.empty(x.shape + (2,))
                    pos[:, :, 0] = x; pos[:, :, 1] = y
                    rv = multivariate_normal(mu[j], sigma[j])
                    ax.contour(x, y, rv.pdf(pos))
                else:
                    
                    x, y = np.mgrid[min(X_pca[:, 0]):max(X_pca[:, 0]):.01, min(X_pca[:, 1]):max(X_pca[:, 1]):.01]
                    pos = np.empty(x.shape + (2,))
                    pos[:, :, 0] = x; pos[:, :, 1] = y
                    V=pca.components_
                    means=pca.mean_
                    rv=multivariate_normal(V.dot(mu[j]-means), V.dot(sigma[j]).dot(V.T))
                    ax.contour(x, y, rv.pdf(pos))
            print(step)

            scatter.set_offsets(X_pca)
            plt.xlabel('Iteration '+str(step))
            plt.pause(0.1)
        




            
