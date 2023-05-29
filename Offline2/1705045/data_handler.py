import numpy as np

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    file = open('data_banknote_authentication.csv', 'r')
    lines = file.readlines()
    file.close()
    X = []
    y = []
    lines=lines[1:]
    for line in lines:
        line = line.strip()
        values = line.split(',')
        values = [float(x) for x in values]
        # prepend 1 to each row
        values.insert(0, 1)
        X.append(values[:-1])
        y.append(values[-1])
    X = np.array(X)
    y = np.array(y)

    # print(X)
    # print(y)


    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """

    assert 0 <= test_size <= 1

    X_train, y_train, X_test, y_test = None, None, None, None
    
    if shuffle:
        order = np.random.permutation(len(X))
        X = X[order]
        y = y[order]
    
    upto = int(len(X) * test_size)
    X_test = X[:upto]
    y_test = y[:upto]
    X_train = X[upto:]
    y_train = y[upto:]

    # print('X_train :' , X_train)
    # print('y_train :' , y_train)
    # print('X_test :' , X_test)
    # print('y_test :' , y_test)
    # print('X_train shape :' , X_train.shape)

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    indices = np.random.randint(0, len(X), len(X))
    X_sample = X[indices]
    y_sample = y[indices]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
