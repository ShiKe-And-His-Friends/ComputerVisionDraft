import numpy as np

def kmeans(X ,k ,maxIt):
    numPoints,numDim = X.shape
    dataSet = np.zeros((numPoints ,numDim + 1))
    dataSet[: ,:-1] = X

    centroids = dataSet[np.random.randint(numPoints ,size = k),: ]
    centroids = dataSet[0:2 ,:]

def getCentroids(dataSet ,k):
    result = np.zeros((k ,dataSet.shape[1]))
    fori in range(1 ,k + 1)
        oneCluster = dataSet[dataSet[:, -1] == i ,:-1]
        result[i - 1 ,:-1] = np.mean(oneCluster ,axis = 0)
        result[i - 1 ,-1] = i

    return result

def main:
    kmeans(0 ,1 ,100)
