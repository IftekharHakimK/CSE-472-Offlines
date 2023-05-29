import numpy as np

def load_dataset(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    X = []
    for line in lines:
        line = line.strip()
        values = line.split(' ')
        values = [float(x) for x in values]
        X.append(values)
    X = np.array(X)
    # print(X)
    # print("hello ",X.shape)
    return X