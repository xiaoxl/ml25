# Input data = (X, y) where X is 2-d numpy array, y is 1-d numpy array.
# Each row is corresponding to a data point.
# Each column is corresponding to a feature.
# The data inX that we want to predict has the same structure as one row in X.
# But inX should be recognized as a 1xn 2-d numpy array.


import numpy as np
# import matplotlib.pyplot as plt


def encodeNorm(X, parameters=None):
    # parameters contains minVals and ranges
    if parameters is None:
        minVals = np.min(X, axis=0)
        maxVals = np.max(X, axis=0)
        ranges = np.maximum(maxVals - minVals, np.ones(minVals.size))
        parameters = {'ranges': ranges, 'minVals': minVals}
    else:
        minVals = parameters['minVals']
        ranges = parameters['ranges']
    Nmat = np.tile(minVals, (X.shape[0], 1))
    Xnorm = (X - Nmat)/ranges
    return (Xnorm, parameters)


def decodeNorm(X, parameters):
    # parameters contains minVals and ranges
    ranges = parameters['ranges']
    minVals = parameters['minVals']
    Nmat = np.tile(minVals, (X.shape[0], 1))
    Xoriginal = X * ranges + Nmat
    return Xoriginal


def classify_kNN(inX, X, y, k):
    # create a new 2-d numpy array by copying inX for each row.
    Xmat = np.tile(np.array([inX]), (X.shape[0], 1))
    # compute the distance between each row of X and Xmat
    Dmat = np.sqrt(np.sum((Xmat - X)**2, axis=1))
    # sort by distance
    sortedlist = Dmat.argsort()
    # count the freq. of the first k items
    k = min(k, len(sortedlist))
    classCount = dict()
    for i in sortedlist[:k]:
        classCount[y[i]] = classCount.get(y[i], 0) + 1
    # find out the most freqent one
    sortedCount = sorted(classCount.items(), key=lambda x:x[1],
                         reverse=True)
    return sortedCount[0][0]


def classify_kNN_test(inTest, outTest, X, y, k):
    n = len(inTest)
    e = 0
    for i in range(n):
        res = classify_kNN(inTest[i], X=X, y=y, k=k)
        if res != outTest[i]:
            e = e + 1
    return e/n


# def dataSplit(X, y, splitrate=.7):
#     N = len(X)
#     n = int(np.floor(N*splitrate))
#     index = list(range(N))
#     np.random.shuffle(index)
#     inTrain = index[:n]
#     inTest = index[n:]
#     X_train = X[inTrain]
#     y_train = y[inTrain]
#     X_test = X[inTest]
#     y_test = y[inTest]
#     return (X_train, y_train, X_test, y_test)



# # %%
# import time

# def createDataSet():
#     group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
#     labels = ['A','A','B','B']
#     return group, labels

# # %%
# inX = np.array([1, 0.9])
# g, l = createDataSet()
# N = 10000
# t0 = time.time()
# for i in range(N):
#     classify_kNN(inX, g, l, 1)
# print(time.time() - t0)

# t0 = time.time()
# for i in range(N):
#     classify_kNN0(inX, g, l, 1)
# print(time.time() - t0)
# # %%
