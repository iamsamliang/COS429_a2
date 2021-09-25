import numpy as np

def kmeans(x, K, niter, seed=123):
    """
    x: array of shape (N, D)
    K: integer
    niter: integer

    centroids: array of shape (K, D)
    labels: array of shape (height*width, )
    """

    np.random.seed(seed)
    idx = np.random.choice(len(x), K, replace=False)

    # Randomly choose centroids
    centroids = x[idx, :]

    # Initialize labels
    labels = np.zeros((x.shape[0], ))



    ### YOUR CODE ###



    return labels, centroids
