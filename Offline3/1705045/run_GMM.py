from data_handler import load_dataset 
from GMM import GMM
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename=input("Filename: ")
    X = load_dataset(filename)
    Ks=[]
    likelihoods=[]

    for k in range(1,11):
        params=dict()
        params['k']=k
        params['iter']=100
        gmm= GMM(params)
        likelihood=gmm.run(X)
        print(k,likelihood)
        Ks.append(k)
        likelihoods.append(likelihood)

    plt.plot(Ks,likelihoods)
    plt.xlabel('K')
    plt.ylabel('Likelihood')
    plt.show()

