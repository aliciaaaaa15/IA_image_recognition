
import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # self.X = np.random.rand(100, 5)

        X = np.asarray(X, dtype=float)

        if X.ndim > 2:
            if X.shape[-1] == 3:
                self.X = X.reshape(-1, 3)


    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = np.zeros((self.K, self.X.shape[1]), dtype=np.float64)
        if self.options['km_init'].lower() == 'first':
            # self.centroids = np.random.rand(self.K, self.X.shape[1])
            # self.old_centroids = np.random.rand(self.K, self.X.shape[1])

            _, list_of_indexes = np.unique(self.X, axis=0, return_index=True)
            sorted_indexes = np.sort(list_of_indexes)
            self.centroids = self.X[sorted_indexes[:self.K]]

        elif self.options['km_init'].lower() == 'random':
        
            rand_elem = np.random.choice(self.K, self.X.shape[1])
            _, list_of_indexes = np.unique(rand_elem, axis=0, return_index=True)
            sorted_indexes = np.sort(list_of_indexes)
            self.centroids = rand_elem[sorted_indexes[:self.K]]
            # list_of_elements = np.unique(rand_elem)
            # list_of_elements = list_of_elements[0:self.K * self.X.shape[1] + 1]
            # self.centroids = np.asarray(list_of_elements, dtype=float)

            print("RANDOM: ", self.centroids)
        elif self.options['km_init'].lower() == 'custom':
            diag_elem = np.diagonal(diag_elem, offset=0, axis1=0, axis2=1)
            _, list_of_indexes = np.unique(diag_elem, axis=0, return_index=True)
            sorted_indexes = np.sort(list_of_indexes)
            self.centroids = diag_elem[sorted_indexes[:self.K]]
            # list_of_elements = np.unique(diag_elem)
            # list_of_elements = list_of_elements[0:self.K * self.X.shape[1] + 1]
            # self.centroids = np.asarray(list_of_elements, dtype=float)

            print("CUSTOM: ", self.centroids)
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])

            print(self.centroids)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #self.labels = np.random.randint(self.K, size=self.X.shape[0])
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.old_centroids = self.centroids.copy()

        for k in range(self.K):
            points = self.X[self.labels == k]  # Get all points assigned to centroid k
            if len(points) > 0:
                self.centroids[k] = np.mean(points, axis=0)  # Compute new centroid
            else:
                # If no points assigned, keep the centroid unchanged (or reinitialize)
                pass  

        # classified_points = {}
        # for index, point in zip(self.labels, self.X):
        #     if index not in classified_points.keys():
        #         classified_points[index] = [point]
        #     else:
        #         classified_points[index].append(point)
        
        # for key in classified_points.keys():
        #     list_elem = np.asarray(classified_points[key], dtype=np.float64)
        #     self.centroids[key] = np.mean(list_elem, axis=0)
        # print("--------------------------------------------")
        # print("Nuevos centroides: ", self.centroids)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # for old, new in zip(self.old_centroids, self.centroids):
        #     if np.linalg.norm(new - old, ord=np.inf, axis=1) <= self.options['tolerance']:
        #     #if np.all(abs(new - old)) <= self.options['tolerance']:
        #         return True
        
        # return self.num_iter >= self.options['max_iter']

        centroid_shift = np.linalg.norm(self.centroids - self.old_centroids, axis=1)
        return np.all(centroid_shift <= self.options['tolerance']) or self.num_iter >= self.options['max_iter']


    def fit(self):
        self._init_centroids()
        while not self.converges():
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        total_distance = 0
        N = self.X.shape[0]  # Number of data points

        for k in range(self.K):
            points = self.X[self.labels == k]  # Get all points assigned to centroid k
            C_k = self.centroids[k]  # Centroid of cluster k
            total_distance += np.sum(np.linalg.norm(points - C_k, axis=1) ** 2)  # Squared distance

        return total_distance / N
        

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        prev_WCD = None
        best_K = max_K  # Default to max_K if no optimal K is found
        threshold = 20  # Fixed threshold as indicated in the image

        for K in range(2, max_K + 1):
            self.K = K
            self.fit()  # Run K-Means with current K
            WCD_k = self.withinClassDistance()  # Compute WCD for current K

            if prev_WCD is not None:
                percent_DEC_k = 100 * (WCD_k / prev_WCD)
                if (100 - percent_DEC_k) < threshold:
                    best_K = K - 1  # Assign previous K as the best K
                    break  # Stop searching once threshold condition is met

            prev_WCD = WCD_k  # Update previous WCD for next iteration

        self.K = best_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    # return np.random.rand(X.shape[0], C.shape[0])
    
    X_expanded = X[:, np.newaxis, :]
    C_expanded = C[np.newaxis, :, :]
    dist = np.sqrt(np.sum((X_expanded - C_expanded) ** 2, axis=2))
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    color_probs = utils.get_color_prob(centroids)  # Kx11 matrix

    # Obtener los índices de los colores con máxima probabilidad
    max_indices = np.argmax(color_probs, axis=1)  # Vector de tamaño K

    # Asignar los nombres de los colores usando utils.colors
    labels = [utils.colors[i] for i in max_indices]

    return labels

    return list(utils.colors)
