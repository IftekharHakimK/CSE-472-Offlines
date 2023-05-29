"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""


def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    tp = 0
    tn = 0
    p = 0
    n = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            p += 1
            if y_pred[i] == 1:
                tp += 1
        else:
            n += 1
            if y_pred[i] == 0:
                tn += 1
    return (tp + tn) / (p + n) * 100
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp = 0
    fp = 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
        else:
            if y_pred[i] == 1:
                fp += 1
    return tp / (tp + fp) * 100


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    tp = 0
    fn = 0
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
            if y_pred[i] == 0:
                fn += 1
    return tp / (tp + fn) * 100


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    
    return 2 * precision_score(y_true, y_pred) * recall_score(y_true, y_pred) / (precision_score(y_true, y_pred) + recall_score(y_true, y_pred))
