from data_handler import load_dataset 
from GMM_with_visualization import GMM
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename=input('Filename: ')
    k_star=int(input("k_star: "))
    
    X = load_dataset(filename)
    
    params=dict()
    params['k']=k_star
    params['iter']=30
    gmm= GMM(params)
    gmm.run(X)
    

