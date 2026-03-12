
import numpy as np
import utils
import time
from scipy.spatial.distance import cdist

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
        self._init_options(options)  # DICT options
        self._init_X(X)



    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        X = np.asarray(X, dtype=np.float64)
        

        if X.ndim > 2 and X.shape[-1] == 3:
            if self.options['filter_border']:
                tolerance = self.options['filter_tolerance']
                border_thickness = self.options['border_thickness']

                top = X[:border_thickness, :, :].reshape(-1, 3)
                bottom = X[-border_thickness:, :, :].reshape(-1, 3)
                left = X[:, :border_thickness, :].reshape(-1, 3)
                right = X[:, -border_thickness:, :].reshape(-1, 3)

                border = np.mean(np.concatenate([top, bottom, left, right], axis=0).reshape(-1, 3), axis=0)
                X_flat = X.reshape(-1, 3)

                diff = np.abs(X_flat - border)
                within_tolerance = np.all(diff <= tolerance, axis=1)

                self.X = X_flat[~within_tolerance]
                self.diag_size = int(np.sqrt(self.X.shape[0]))
            else:
                self.diag_size=min(X.shape[0], X.shape[1])
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
            options['km_init'] = 'first' # Values: first, diagonal, random, kmeans++
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # Values: WCD, BCD, Fisher, Fisher_Elbow
        if 'threshold_type' not in options:
            options['threshold_type'] = 'static' # Values: static, dynamic, decay-aware
        if 'decay_threshold' not in options:
            options['decay_threshold'] = 0.01
        if 'filter_border' not in options:
            options['filter_border'] = False
        if 'border_thickness' not in options:
            options['border_thickness'] = 5
        if 'filter_tolerance' not in options:
            options['filter_tolerance'] = 22 # Best value


        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options


    def _init_centroids(self):
        """
        Initialization of centroids
        """
        self.old_centroids = np.zeros((self.K, self.X.shape[1]), dtype=np.float64)

        if self.options['km_init'].lower() == 'first':
            _, list_of_indexes = np.unique(self.X, axis=0, return_index=True)
            sorted_indexes = np.sort(list_of_indexes)
            self.centroids = self.X[sorted_indexes[:self.K]]

        elif self.options['km_init'].lower() == 'random':
            np.random.seed(int(time.time_ns() % (2**32)))
            rand_index = np.random.randint(0, self.X.shape[0], size=self.K)
            self.centroids = self.X[rand_index]
            self.initial_centroid_index = rand_index

        elif self.options['km_init'].lower() == 'random_choice':
            rand_index = np.random.choice(self.X.shape[0], self.K)
            self.centroids = self.X[rand_index]

        elif self.options['km_init'].lower() == 'diagonal': # Not suitable for filtered images
            diag_index = [x * (self.diag_size + 1) for x in range(0, self.diag_size, int(self.diag_size / self.K))]
            self.centroids = self.X[diag_index[:self.K]]

        elif self.options['km_init'].lower() == 'kmeans++':
            self.centroids = np.empty((self.K, self.X.shape[1]), dtype=np.float64)
            np.random.seed(int(time.time_ns() % (2**32)))
            self.centroids[0] = self.X[np.random.randint(self.X.shape[0])]

            for k in range(1, self.K):
                dists = np.min(cdist(self.X, self.centroids[:k]) ** 2, axis=1)
                probs = dists / np.sum(dists)
                cumulative_probs = np.cumsum(probs)
                r = np.random.rand()
                next_index = np.searchsorted(cumulative_probs, r)
                self.centroids[k] = self.X[next_index]
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])


    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids.copy()
        channels = self.X.shape[1] 
        counts = np.bincount(self.labels, minlength=self.K).astype(np.float64)

        new_centroids = np.zeros((self.K, channels), dtype=np.float64)

        # Sum of points per cluster for each dimension
        for channel in range(channels):
            sums = np.bincount(self.labels, weights=self.X[:, channel], minlength=self.K)
            new_centroids[:, channel] = np.divide(sums, counts, out=np.zeros(self.K), where=counts != 0)

        self.centroids = new_centroids

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """

        centroid_shift = np.linalg.norm(self.centroids - self.old_centroids, axis=1)
        return np.all(centroid_shift <= self.options['tolerance']) or self.num_iter >= self.options['max_iter']


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self._init_centroids()
        while not self.converges():
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
        

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        total_distance = 0

        for k in range(self.K):
            points = self.X[self.labels == k]
            C_k = self.centroids[k]
            total_distance += np.sum(np.linalg.norm(points - C_k, axis=1) ** 2)

        return total_distance / self.X.shape[0]
        

    def interClassDistance(self):
        """
         returns the between class distance of the current clustering
        """
        overall_mean = np.mean(self.X, axis=0)
        total_distance = 0

        for k in range(self.K):
            n_k = np.sum(self.labels == k)  # 1 divided by the number of points in cluster k
            C_k = self.centroids[k]
            total_distance += n_k * (np.linalg.norm(C_k - overall_mean) ** 2)

        return total_distance

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        prev_CD = None
        prev_improvement = None
        current_improvement = None
        best_K = max_K
        llindar = 20
        min_threshold = 5

        if self.options['fitting'] == 'WCD':
            for K in range(2, max_K + 1):
                self.K = K
                self.fit()
                WCD_k = self.withinClassDistance()

                if prev_CD is not None:
                    percent_DEC_k = 100 * (WCD_k / prev_CD)

                    if (100 - percent_DEC_k) < llindar:
                        best_K = K - 1
                        break

                prev_CD = WCD_k
        elif self.options['fitting'] == 'BCD':
            for K in range(2, max_K + 1):
                self.K = K
                self.fit()
                BCD_k = self.interClassDistance()

                if prev_CD is not None:
                    
                    percent_INC_k = 100 * ((BCD_k - prev_CD) / prev_CD)
                    if percent_INC_k < llindar:
                        best_K = K - 1
                        break

                prev_CD = BCD_k
        elif self.options['fitting'] == 'Fisher': #BCD/WCD Ratio
            for K in range(2, max_K + 1):
                self.K = K
                best_score = -np.inf
                self.fit()

                WCD_k = self.withinClassDistance()
                BCD_k = self.interClassDistance()

                if WCD_k == 0:
                    continue  # we dont want to divide by zero!

                Fisher_C = BCD_k / WCD_k

                if prev_CD == 0:
                    continue
                
                if self.options['threshold_type'] == 'static':
                    if prev_CD is not None:
                        percent_INC_R = 100 * ((Fisher_C - prev_CD) / prev_CD)
                        if percent_INC_R < llindar:
                            break

                if self.options['threshold_type'] == 'dynamic':
                    if prev_CD is not None:
                        current_improvement = (Fisher_C - prev_CD) / prev_CD

                    if prev_CD is not None and prev_improvement is not None:
                        decay = abs(current_improvement - prev_improvement)
                    
                        if decay < self.options['decay_threshold']: 
                            break

                    if current_improvement is not None:
                        prev_improvement = current_improvement

                if self.options['threshold_type'] == 'decay-aware':
                    if prev_CD is not None:
                        current_improvement = 100 * ((Fisher_C - prev_CD) / prev_CD)

                        if prev_improvement is not None:
                            improvement_decay = abs(current_improvement - prev_improvement)

                            if current_improvement < min_threshold and improvement_decay < self.options['decay_threshold']:
                                break

                    if current_improvement is not None:
                        prev_improvement = current_improvement

                if Fisher_C > best_score:
                    best_score = Fisher_C
                    best_K = K
                
                prev_CD = Fisher_C
        elif self.options['fitting'] == 'Fisher_Elbow':
            Ks = list(range(2, max_K + 1))
            Fisher_Cs = []

            for K in Ks:
                self.K = K
                self.fit()

                WCD_k = self.withinClassDistance()
                BCD_k = self.interClassDistance()

                if WCD_k == 0:
                    Fisher_Cs.append(0)
                else:
                    Fisher_Cs.append(BCD_k / WCD_k)

            # Convert K and scores to numpy arrays
            ks = np.array(Ks)
            ys = np.array(Fisher_Cs)

            # Line from first to last point
            p1 = np.array([ks[0], ys[0]])
            p2 = np.array([ks[-1], ys[-1]])

            # Vector of the line
            line_vec = p2 - p1
            line_vec_norm = line_vec / np.linalg.norm(line_vec)

            # Compute perpendicular distances to the line
            distances = []
            for i in range(len(ks)):
                p = np.array([ks[i], ys[i]])
                vec_to_line = p - p1
                proj_len = np.dot(vec_to_line, line_vec_norm)
                proj_point = p1 + proj_len * line_vec_norm
                distance = np.linalg.norm(p - proj_point)
                distances.append(distance)

            # Elbow is at the point of maximum distance
            best_index = int(np.argmax(distances))
            best_K = ks[best_index]

        self.K = best_K

    def get_dominant_colors(self, min_percentage = 0.1):

        total_pixels = len(self.labels)
        frequency = np.bincount(self.labels, minlength=self.K)

        color_map = {}
        for i, count in enumerate(frequency):
            percent = count / total_pixels
            centroid = self.centroids[i].reshape(1, -1)
            color_name = get_colors(centroid)[0]  # retorna un sol color
            color_map[color_name] = color_map.get(color_name, 0) + percent

        dominant_colors = [
        (color, round(p, 2))
        for color, p in color_map.items()
        if p >= min_percentage
        ]
        dominant_colors.sort(key=lambda x: x[1], reverse=True)

        return dominant_colors


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
    return cdist(X, C, metric='euclidean')


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    color_probs = utils.get_color_prob(centroids)
    max_index = np.argmax(color_probs, axis=1)
    return [utils.colors[i] for i in max_index]



