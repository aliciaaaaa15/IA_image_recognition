
import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        train_data = train_data.astype(float)
        P = train_data.shape[0]
        self.train_data = train_data.reshape(P, -1)



    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates the k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        test_data = test_data.astype(float)
        P = test_data.shape[0]
        test_data = test_data.reshape(P, -1)

   
        distancies = cdist(test_data, self.train_data)
        index_veins = np.argsort(distancies, axis=1)[:, :k]
        self.neighbors = self.labels[index_veins]

    def get_class(self):

        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """


        frequents = []
        for row in self.neighbors:
            _, index, counts = np.unique(row, return_index=True, return_counts=True)
            frequents.append(row[min(index[counts == max(counts)])])
        
        return np.array(frequents)



    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
