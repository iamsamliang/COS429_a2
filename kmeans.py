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
    for i in range(niter):
        # for each pixel subtract the RGB values of that pixel with the RGB values of each centroid and take the L2 norm
        dist = np.linalg.norm(centroids-x[:,np.newaxis], axis = 2)
        # assign the pixel to the closest centroid based on this L2 norm value
        labels = np.argmin(dist, axis=1) # labels[0] = 1 means pixel in row 0 of x is closest to centroid 1 (row 1 of centroids)
        # recalculate new centroids
        for centroid in range(centroids.shape[0]):
            x_assoc_centroid = np.where(labels==centroid)[0] # return the index of the pixels associated with this centroid
            centroids[centroid] = np.mean(x[x_assoc_centroid], axis=0) # add each component and take its mean


    return labels, centroids
